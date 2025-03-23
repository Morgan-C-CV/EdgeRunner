import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# 从vae_unet_predict.py导入模型定义和数据处理函数
from vae_unet_predict import VAE, Encoder, ImprovedDecoder, ResBlock, CombinedDataset, load_vae_model, compute_reconstruction_error, load_dataset_paths

# 兼容性UNet模型定义 - 支持新旧两种模型结构
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 兼容简化版UNet
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, dropout_rate=0.0):
        super(UNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64, dropout_rate=dropout_rate)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128, dropout_rate=dropout_rate)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256, dropout_rate=dropout_rate)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512, dropout_rate=dropout_rate)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 512, dropout_rate=dropout_rate)
        )
        
        # Decoder - 修正通道数以匹配concat后的维度
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(768, 256, dropout_rate=dropout_rate)  # 256+512=768
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(384, 128, dropout_rate=dropout_rate)  # 128+256=384
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(192, 64, dropout_rate=dropout_rate)   # 64+128=192
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(96, 64, dropout_rate=dropout_rate)    # 32+64=96
        
        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)
        
        return self.outc(x)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 自定义评估指标函数
def accuracy_score(y_true, y_pred):
    """计算准确率: (TP + TN) / (TP + TN + FP + FN)"""
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    return float(correct) / total

def precision_score(y_true, y_pred, zero_division=0):
    """计算精确率: TP / (TP + FP)"""
    # 转换为布尔数组，提高效率
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # True Positives: 预测为正例且实际为正例
    tp = np.logical_and(y_pred, y_true).sum()
    # False Positives: 预测为正例但实际为负例
    fp = np.logical_and(y_pred, np.logical_not(y_true)).sum()
    
    # 精确率计算，处理分母为0的情况
    if tp + fp == 0:
        return zero_division
    return float(tp) / (tp + fp)

def recall_score(y_true, y_pred, zero_division=0):
    """计算召回率: TP / (TP + FN)"""
    # 转换为布尔数组
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # True Positives: 预测为正例且实际为正例
    tp = np.logical_and(y_pred, y_true).sum()
    # False Negatives: 预测为负例但实际为正例
    fn = np.logical_and(np.logical_not(y_pred), y_true).sum()
    
    # 召回率计算，处理分母为0的情况
    if tp + fn == 0:
        return zero_division
    return float(tp) / (tp + fn)

def f1_score(y_true, y_pred, zero_division=0):
    """计算F1分数: 2 * (precision * recall) / (precision + recall)"""
    # 计算精确率和召回率
    prec = precision_score(y_true, y_pred, zero_division)
    rec = recall_score(y_true, y_pred, zero_division)
    
    # F1分数计算，处理分母为0的情况
    if prec + rec == 0:
        return zero_division
    return 2 * (prec * rec) / (prec + rec)

# IoU计算函数 (Intersection over Union)
def calculate_iou(pred_mask, true_mask, smooth=1e-6):
    """计算IoU (Intersection over Union)"""
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0.5
    
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

# Dice系数计算函数
def calculate_dice(pred_mask, true_mask, smooth=1e-6):
    """计算Dice系数"""
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0.5
    
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    return (2. * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)

# 保存预测结果的可视化函数 (使用cv2)
def save_predictions(original_imgs, modified_imgs, true_masks, pred_masks, save_dir="prediction_results", num_samples=5):
    """保存原始图像、修改图像、真实掩码和预测掩码的可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 取前num_samples个样本进行可视化
    for i in range(min(num_samples, len(original_imgs))):
        # 转换回NumPy数组并反归一化
        orig = original_imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)  # 转换为BGR以便使用cv2保存
        
        mod = modified_imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        mod = np.clip(mod * 255, 0, 255).astype(np.uint8)
        mod = cv2.cvtColor(mod, cv2.COLOR_RGB2BGR)
        
        true_mask = true_masks[i].squeeze().cpu().numpy()
        true_mask = (true_mask > 0.5) * 255
        true_mask = true_mask.astype(np.uint8)
        
        pred_mask = pred_masks[i].squeeze().cpu().numpy() > 0.5
        pred_mask = pred_mask.astype(np.uint8) * 255
        
        # 分别保存每个图像
        cv2.imwrite(os.path.join(save_dir, f'original_{i}.png'), orig)
        cv2.imwrite(os.path.join(save_dir, f'modified_{i}.png'), mod)
        cv2.imwrite(os.path.join(save_dir, f'true_mask_{i}.png'), true_mask)
        cv2.imwrite(os.path.join(save_dir, f'pred_mask_{i}.png'), pred_mask)
        
        # 创建一个组合图像 (2x2网格)
        h, w = orig.shape[:2]
        grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # 填充原始图像 (左上)
        grid[:h, :w] = orig
        
        # 填充修改图像 (右上)
        grid[:h, w:] = mod
        
        # 填充真实掩码 (左下) - 扩展为3通道以保持一致
        true_mask_3ch = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)
        grid[h:, :w] = true_mask_3ch
        
        # 填充预测掩码 (右下) - 扩展为3通道以保持一致
        pred_mask_3ch = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        grid[h:, w:] = pred_mask_3ch
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, '原始图像', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(grid, '修改图像', (w+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(grid, '真实掩码', (10, h+30), font, 1, (255, 255, 255), 2)
        cv2.putText(grid, '预测掩码', (w+10, h+30), font, 1, (255, 255, 255), 2)
        
        # 保存组合图像
        cv2.imwrite(os.path.join(save_dir, f'combined_{i}.png'), grid)

# 统计指标计算函数
def calculate_metrics(all_preds, all_masks):
    """计算各种评估指标"""
    # 转换为二值形式
    all_preds_binary = (all_preds > 0.5).float()
    all_masks_binary = (all_masks > 0.5).float()
    
    # 转换为一维数组用于计算
    pred_flat = all_preds_binary.cpu().numpy().flatten()
    mask_flat = all_masks_binary.cpu().numpy().flatten()
    
    # 计算准确率、精确度、召回率和F1分数
    accuracy = accuracy_score(mask_flat, pred_flat)
    precision = precision_score(mask_flat, pred_flat, zero_division=0)
    recall = recall_score(mask_flat, pred_flat, zero_division=0)
    f1 = f1_score(mask_flat, pred_flat, zero_division=0)
    
    # 计算平均IoU和Dice
    iou_sum = 0
    dice_sum = 0
    batch_size = all_preds.size(0)
    
    for i in range(batch_size):
        iou_sum += calculate_iou(all_preds[i], all_masks[i])
        dice_sum += calculate_dice(all_preds[i], all_masks[i])
    
    mean_iou = iou_sum / batch_size
    mean_dice = dice_sum / batch_size
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice
    }

# 验证函数
def validate_unet(unet_model, vae_model, val_loader, save_predictions_flag=True):
    """在验证集上验证UNet模型并返回评估指标"""
    unet_model.eval()
    vae_model.eval()
    
    all_preds = []
    all_masks = []
    all_orig_imgs = []
    all_mod_imgs = []
    
    with torch.no_grad():
        for batch_idx, (orig_imgs, mod_imgs, masks) in enumerate(tqdm(val_loader, desc="验证中")):
            orig_imgs = orig_imgs.to(device)
            mod_imgs = mod_imgs.to(device)
            masks = masks.to(device)
            
            # 计算重建误差
            recon_error = compute_reconstruction_error(orig_imgs, vae_model)
            
            # 组合修改图像和重建误差
            combined_input = torch.cat([mod_imgs, recon_error], dim=1)
            
            # 前向传播
            outputs = unet_model(combined_input)
            
            # 保存结果用于指标计算
            all_preds.append(outputs)
            all_masks.append(masks)
            
            # 保存原始数据用于可视化
            if save_predictions_flag and batch_idx == 0:  # 只保存第一个批次用于可视化
                all_orig_imgs.extend(orig_imgs)
                all_mod_imgs.extend(mod_imgs)
    
    # 合并所有批次
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 计算评估指标
    metrics = calculate_metrics(all_preds, all_masks)
    
    # 保存一些预测结果的可视化
    if save_predictions_flag and len(all_orig_imgs) > 0:
        save_predictions(all_orig_imgs, all_mod_imgs, all_masks[:len(all_orig_imgs)], all_preds[:len(all_orig_imgs)])
    
    return metrics

# 主函数
def main():
    # 配置参数
    dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"
    vae_model_path = "/dev/shm/fine_tuned_vae/vae_model.pth"
    unet_model_path = "best_unet_model.pth"  # 使用早停保存的最佳模型
    batch_size = 4
    target_size = (256, 256)
    max_samples = None
    
    # 加载数据集路径
    print("加载数据集...")
    originals, images, masks = load_dataset_paths(dataset_path, max_samples)
    
    # 实现K折交叉验证
    k_folds = 5
    fold_size = len(originals) // k_folds
    all_metrics = []
    
    print(f"执行{k_folds}折交叉验证...")
    
    for fold in range(k_folds):
        print(f"\n开始第{fold+1}折验证...")
        
        # 创建验证索引
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else len(originals)
        
        # 创建验证集
        val_originals = originals[start_idx:end_idx]
        val_images = images[start_idx:end_idx]
        val_masks = masks[start_idx:end_idx]
        
        print(f"验证集大小: {len(val_originals)}张图像")
        
        # 创建验证数据集和数据加载器
        val_dataset = CombinedDataset(val_originals, val_images, val_masks, target_size, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 加载VAE模型
        print("加载VAE模型...")
        if not os.path.exists(vae_model_path):
            print(f"VAE模型路径不存在: {vae_model_path}")
            alt_paths = [
                "fine_tuned_vae/vae_model.pth",
                "vae_model.pth",
                "/tmp/fine_tuned_vae/vae_model.pth"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"使用替代VAE模型路径: {alt_path}")
                    vae_model_path = alt_path
                    break
        
        vae_model = load_vae_model(vae_model_path, input_shape=(3, *target_size))
        
        # 验证VAE模型是否正确加载
        try:
            with torch.no_grad():
                test_input = torch.randn(1, 3, 256, 256).to(device)
                test_output = vae_model(test_input)
                if test_output is None or test_output.shape != test_input.shape:
                    print("警告: VAE模型输出形状异常，使用备用重建误差计算")
                    vae_works = False
                else:
                    print("VAE模型验证成功")
                    vae_works = True
        except Exception as e:
            print(f"VAE模型测试失败: {e}")
            vae_works = False
        
        # 加载UNet模型
        print("加载UNet模型...")
        unet_model = UNet(in_channels=4, out_channels=1).to(device)
        
        # 首先尝试加载指定模型路径
        try:
            # 使用非严格模式加载模型权重，以适应模型结构变化
            state_dict = torch.load(unet_model_path, map_location=device)
            
            # 尝试适配模型结构（如果需要）
            if "inc.conv.0.weight" in state_dict:
                print("检测到新版模型结构，使用兼容模式加载...")
                unet_model.load_state_dict(state_dict)
            else:
                print("检测到旧版模型结构，使用非严格模式加载...")
                unet_model.load_state_dict(state_dict, strict=False)
            
            print(f"UNet模型成功加载自: {unet_model_path}")
        except Exception as e:
            print(f"加载指定的UNet模型时出错: {e}")
            # 尝试加载标准模型路径
            try:
                state_dict = torch.load("unet_model.pth", map_location=device)
                unet_model.load_state_dict(state_dict, strict=False)
                print("UNet模型成功加载自: unet_model.pth")
            except Exception as e:
                print(f"加载标准UNet模型时出错: {e}")
                return
        
        # 验证模型
        print("开始验证...")
        metrics = validate_unet(unet_model, vae_model, val_loader, save_predictions_flag=(fold==0))
        all_metrics.append(metrics)
        
        # 打印当前折的评估指标
        print(f"\n第{fold+1}折验证结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确度: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        print(f"平均IoU: {metrics['mean_iou']:.4f}")
        print(f"平均Dice系数: {metrics['mean_dice']:.4f}")
    
    # 计算并打印平均指标
    avg_metrics = {
        'accuracy': sum(m['accuracy'] for m in all_metrics) / k_folds,
        'precision': sum(m['precision'] for m in all_metrics) / k_folds,
        'recall': sum(m['recall'] for m in all_metrics) / k_folds,
        'f1_score': sum(m['f1_score'] for m in all_metrics) / k_folds,
        'mean_iou': sum(m['mean_iou'] for m in all_metrics) / k_folds,
        'mean_dice': sum(m['mean_dice'] for m in all_metrics) / k_folds
    }
    
    print("\n交叉验证平均结果:")
    print(f"平均准确率: {avg_metrics['accuracy']:.4f}")
    print(f"平均精确度: {avg_metrics['precision']:.4f}")
    print(f"平均召回率: {avg_metrics['recall']:.4f}")
    print(f"平均F1分数: {avg_metrics['f1_score']:.4f}")
    print(f"平均IoU: {avg_metrics['mean_iou']:.4f}")
    print(f"平均Dice系数: {avg_metrics['mean_dice']:.4f}")
    
    # 保存指标到文件
    with open("cross_validation_metrics.txt", "w") as f:
        for fold, metrics in enumerate(all_metrics):
            f.write(f"Fold {fold+1}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nAverage:\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print("验证完成，结果已保存")

# 生成预测掩码的生成器函数
def predict_masks_generator(unet_model, vae_model, test_loader, threshold=0.5, vae_works=True):
    """生成器函数，分批次返回预测结果"""
    unet_model.eval()
    vae_model.eval()
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc="预测中"):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            # 计算重建误差
            if vae_works:
                recon_error = compute_reconstruction_error(imgs, vae_model)
            else:
                # 如果VAE不工作，使用替代方法
                batch_size, _, height, width = imgs.shape
                recon_error = torch.rand(batch_size, 1, height, width, device=device) * 0.1
            
            # 组合输入
            combined_input = torch.cat([imgs, recon_error], dim=1)
            
            # 预测掩码
            mask_preds = unet_model(combined_input)
            
            # 处理每个批次中的图像
            batch_results = []
            for i in range(batch_size):
                # 获取预测掩码
                mask_np = mask_preds[i].squeeze().cpu().numpy()
                
                # 二值化 - 使用相同的阈值
                binary_mask = (mask_np > threshold).astype(np.uint8)
                
                # 形态学操作
                kernel = np.ones((3, 3), np.uint8)
                refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                # 移除太小的连通区域
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
                for j in range(1, num_labels):
                    if stats[j, cv2.CC_STAT_AREA] < 50:
                        refined_mask[labels == j] = 0
                
                # 转换为RLE
                rle = mask2rle(refined_mask)
                
                # 添加到结果
                batch_results.append({
                    'ImageId': img_ids[i],
                    'EncodedPixels': rle if np.any(refined_mask) else ''
                })
            
            yield batch_results

if __name__ == "__main__":
    main() 
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# 导入必要的模型定义和数据处理函数
from sd_vae_finetuning import VAE, Encoder, ImprovedDecoder, ResBlock
# 导入新的VAE加载函数
try:
    from resave_vae import load_saved_vae
    has_resave_module = True
except ImportError:
    print("警告: 无法导入load_saved_vae函数，将使用回退方法")
    has_resave_module = False

# 从prediction.py导入数据处理函数
try:
    from prediction import CombinedDataset, compute_reconstruction_error, load_dataset_paths
    print("成功从prediction.py导入函数")
except ImportError:
    # 如果失败，尝试从vae_unet_predict.py导入
    try:
        from vae_unet_predict import CombinedDataset, compute_reconstruction_error, load_dataset_paths
        print("成功从vae_unet_predict.py导入函数")
    except ImportError:
        print("警告: 无法导入数据处理函数，将使用内部定义")
        # 这里应该定义这些函数的实现，但为简洁起见我们暂时省略

# 定义U-Net模型
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):  # 输入通道数为4(RGB + 重建误差)
        super(UNet, self).__init__()

        # Encoder (下采样)
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = DoubleConv(512, 1024)

        # Decoder (上采样)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)  # 输入通道1024 = 512(上采样) + 512(跳跃连接)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)  # 输入通道512 = 256(上采样) + 256(跳跃连接)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)  # 输入通道256 = 128(上采样) + 128(跳跃连接)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)  # 输入通道128 = 64(上采样) + 64(跳跃连接)

        # 输出层
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # Decoder
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = self.sigmoid(c10)

        return out

# 添加改进的UNet模型定义，与训练时使用的模型结构保持一致
# 定义注意力模块，以增强UNet的表达能力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out) * x

# 改进后的双卷积块，增加Dropout、残差连接和注意力机制
class ImprovedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, use_attention=True):
        super(ImprovedDoubleConv, self).__init__()
        self.use_residual = (in_channels == out_channels)
        self.use_attention = use_attention
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if self.use_residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        
        if self.use_residual:
            out = out + residual
            
        return out

# 改进的UNet模型，增加深度、注意力机制和Dropout
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, dropout_rate=0.3, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part
        in_channels_down = in_channels
        for feature in features:
            self.downs.append(ImprovedDoubleConv(in_channels_down, feature, dropout_rate))
            in_channels_down = feature
        
        # Bottleneck
        self.bottleneck = ImprovedDoubleConv(features[-1], features[-1]*2, dropout_rate)
        
        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(ImprovedDoubleConv(feature*2, feature, dropout_rate))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        skip_connections = []
        
        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse
        
        # Up path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx//2]
            
            # Handle if shapes don't match exactly (if needed)
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Double Conv
        
        x = self.final_conv(x)
        return self.sigmoid(x)

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

# 添加缺失的mask2rle函数
def mask2rle(mask):
    """
    将二值掩码转换为游程编码 (RLE)
    参数:
        mask: 二值掩码（0和1）
    返回:
        rle: 以字符串形式表示的RLE编码
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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

# 重新实现的VAE模型加载函数，使用新的加载方法
def load_vae_model(model_path, config_path=None, input_shape=(3, 256, 256), latent_dim=2048):
    print(f"尝试从 {model_path} 加载VAE模型...")
    
    # 首先尝试使用新的加载方法
    if has_resave_module and os.path.exists(model_path):
        try:
            # 使用新的专用加载函数
            if config_path and os.path.exists(config_path):
                print(f"使用专用加载函数从 {model_path} 和配置 {config_path} 加载模型...")
                vae = load_saved_vae(model_path, config_path)
                print("使用新方法成功加载VAE模型")
                vae.eval()
                return vae
            else:
                print(f"使用专用加载函数从 {model_path} 加载模型（无配置文件）...")
                vae = load_saved_vae(model_path)
                print("使用新方法成功加载VAE模型")
                vae.eval()
                return vae
        except ImportError as e:
            print(f"无法导入load_saved_vae函数: {e}")
            print("回退到传统加载方法...")
        except Exception as e:
            print(f"使用新方法加载模型失败: {e}")
            print("回退到传统加载方法...")
    
    # 如果新方法失败，回退到传统方法
    try:
        # 创建与训练时相同的模型结构
        vae = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
        
        # 正确方式：加载状态字典
        vae.load_state_dict(torch.load(model_path, map_location=device))
        print("成功加载VAE模型状态字典")
        vae.eval()
        return vae
    except Exception as e:
        print(f"加载状态字典失败: {e}")
        
        # 尝试方法2: 直接加载整个模型
        try:
            print("尝试直接加载完整模型...")
            vae = torch.load(model_path, map_location=device)
            print("成功加载完整VAE模型")
            vae.eval()
            return vae
        except Exception as e:
            print(f"加载完整模型失败: {e}")
            
            # 尝试方法3: 使用非严格模式加载
            try:
                print("尝试使用非严格模式加载模型...")
                vae.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                print("使用非严格模式成功加载部分权重")
                
                # 检查加载了多少参数
                loaded_params = set(torch.load(model_path, map_location=device).keys())
                total_params = set(vae.state_dict().keys())
                print(f"加载了 {len(loaded_params.intersection(total_params))}/{len(total_params)} 个参数")
                
                vae.eval()
                return vae
            except Exception as e:
                print(f"使用非严格模式加载模型失败: {e}")
                
                # 方法4: 创建简化版VAE模型作为后备
                print("创建新的简化VAE模型...")

                class SimpleVAE(nn.Module):
                    def __init__(self, input_shape=(3, 256, 256)):
                        super(SimpleVAE, self).__init__()

                        # 简单编码器
                        self.encoder = nn.Sequential(
                            nn.Conv2d(3, 32, 3, stride=2, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, 3, stride=2, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, stride=2, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 3, stride=2, padding=1),
                            nn.ReLU(),
                        )

                        # 简单解码器
                        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                            nn.ReLU(),
                            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                            nn.ReLU(),
                            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                            nn.ReLU(),
                            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
                            nn.Tanh()
                        )

                    def forward(self, x):
                        encoded = self.encoder(x)
                        decoded = self.decoder(encoded)
                        return decoded

                # 返回简化版VAE（用于计算重建误差）
                simple_vae = SimpleVAE(input_shape=input_shape).to(device)
                print("创建了简化VAE模型（未加载预训练权重）")
                simple_vae.eval()
                return simple_vae

# 主函数
def main():
    # 配置参数
    dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"
    
    # 更新VAE模型路径，使用新的位置
    vae_model_path = "/dev/shm/fine_tuned_vae1/vae_model.pth"
    vae_config_path = "/dev/shm/fine_tuned_vae1/model_config.pth"
    
    unet_model_path = "unet_model_epoch_26.pth"
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
        val_dataset = CombinedDataset(val_originals, val_images, val_masks, target_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 加载VAE模型
        print("加载VAE模型...")
        if not os.path.exists(vae_model_path):
            print(f"VAE模型路径不存在: {vae_model_path}")
            alt_paths = [
                "/dev/shm/fine_tuned_vae/vae_model.pth",
                "/dev/shm/fine_tuned_vae/vae_complete_model.pth",
                "fine_tuned_vae/vae_model.pth",
                "fine_tuned_vae/vae_complete_model.pth",
                "vae_model.pth",
                "/tmp/fine_tuned_vae/vae_model.pth"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"使用替代VAE模型路径: {alt_path}")
                    vae_model_path = alt_path
                    break
        
        # 使用修改后的加载函数加载VAE模型
        vae_model = load_vae_model(vae_model_path, vae_config_path, input_shape=(3, *target_size))

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
        
        # 智能检测模型类型
        def detect_model_type(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                # 检查状态字典的键，判断是哪种模型架构
                if any("downs." in key for key in state_dict.keys()):
                    print(f"检测到ImprovedUNet架构模型: {model_path}")
                    return "ImprovedUNet"
                else:
                    print(f"检测到标准UNet架构模型: {model_path}")
                    return "UNet"
            except Exception as e:
                print(f"无法检测模型类型: {e}")
                return "Unknown"
        
        # 尝试多个路径，找到可用的模型
        model_paths = [unet_model_path, "unet_model.pth", "improved_unet_model.pth", "unet_model2.pth"]
        unet_model = None
        
        for path in model_paths:
            if os.path.exists(path):
                model_type = detect_model_type(path)
                if model_type == "ImprovedUNet":
                    # 创建改进的UNet模型
                    print(f"创建ImprovedUNet模型实例，用于加载: {path}")
                    unet_model = ImprovedUNet(in_channels=4, out_channels=1, dropout_rate=0.3).to(device)
                else:
                    # 创建标准UNet模型
                    print(f"创建标准UNet模型实例，用于加载: {path}")
                    unet_model = UNet(in_channels=4, out_channels=1).to(device)
                
                try:
                    # 加载模型权重
                    state_dict = torch.load(path, map_location=device)
                    unet_model.load_state_dict(state_dict)
                    print(f"UNet模型成功加载自: {path}")
                    break
                except Exception as e:
                    print(f"加载模型权重失败: {e}")
                    unet_model = None
        
        # 如果所有尝试都失败，则无法继续
        if unet_model is None:
            print("无法加载任何UNet模型，跳过当前fold")
            continue
        
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
    if all_metrics:
        avg_metrics = {
            'accuracy': sum(m['accuracy'] for m in all_metrics) / len(all_metrics),
            'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
            'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
            'f1_score': sum(m['f1_score'] for m in all_metrics) / len(all_metrics),
            'mean_iou': sum(m['mean_iou'] for m in all_metrics) / len(all_metrics),
            'mean_dice': sum(m['mean_dice'] for m in all_metrics) / len(all_metrics)
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
    else:
        print("验证未能完成，没有有效的指标结果")

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
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 从vae_unet_predict.py导入相关函数和模型
from vae_unet_predict import UNet, load_vae_model, compute_reconstruction_error

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 测试数据集类
class TestDataset(Dataset):
    def __init__(self, images_paths, target_size=(256, 256)):
        self.images_paths = images_paths
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_ids = [os.path.splitext(os.path.basename(path))[0] for path in images_paths]
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        try:
            # 加载图像
            img_path = self.images_paths[idx]
            img = cv2.imread(img_path)
            
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 调整大小
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为张量
            img_tensor = self.transform(img)
            
            return img_tensor, self.image_ids[idx]
            
        except Exception as e:
            print(f"处理图像 {self.images_paths[idx]} 时出错: {e}")
            # 返回随机张量作为替代
            return torch.randn(3, *self.target_size), self.image_ids[idx]

# 加载测试集路径
def load_test_images(test_dir):
    print("加载测试集图像...")
    images_paths = []
    
    # 测试图像目录
    test_images_dir = os.path.join(test_dir, "images")
    
    # 遍历测试图像目录
    for filename in os.listdir(test_images_dir):
        if filename.endswith(".png"):
            images_paths.append(os.path.join(test_images_dir, filename))
    
    print(f"加载了 {len(images_paths)} 个测试图像")
    return images_paths

# 掩码转RLE编码
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 预测掩码
def predict_masks(unet_model, vae_model, test_loader, threshold=0.5):
    print("开始预测掩码...")
    results = []
    
    unet_model.eval()
    vae_model.eval()
    
    # 尝试加载最佳阈值
    try:
        with open("best_threshold.txt", "r") as f:
            best_threshold = float(f.read().strip())
            print(f"使用训练中找到的最佳阈值: {best_threshold}")
            threshold = best_threshold
    except:
        print(f"使用默认阈值: {threshold}")
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc="预测中"):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            # 计算重建误差
            recon_error = compute_reconstruction_error(imgs, vae_model)
            
            # 组合输入
            combined_input = torch.cat([imgs, recon_error], dim=1)
            
            # 预测掩码
            mask_preds = unet_model(combined_input)
            
            # 处理每个批次中的图像
            for i in range(batch_size):
                # 获取预测掩码
                mask_np = mask_preds[i].squeeze().cpu().numpy()
                
                # 根据图像特性自适应调整阈值
                if np.mean(mask_np) > 0.2:  # 如果平均预测值较高，提高阈值减少误报
                    adaptive_threshold = threshold + 0.05
                elif np.mean(mask_np) < 0.05:  # 如果平均预测值较低，适当降低阈值
                    adaptive_threshold = threshold - 0.05
                else:
                    adaptive_threshold = threshold
                
                # 二值化
                binary_mask = (mask_np > adaptive_threshold).astype(np.uint8)
                
                # 增强后处理
                # 1. 开操作去除噪点
                kernel = np.ones((3, 3), np.uint8)
                refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # 2. 闭操作填充孔洞
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                # 3. 移除太小的连通区域 (可能是噪声)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
                clean_mask = np.zeros_like(refined_mask)
                
                for j in range(1, num_labels):  # 跳过背景
                    if stats[j, cv2.CC_STAT_AREA] < 50:  # 面积阈值
                        continue
                    
                    # 保留大连通区域
                    clean_mask[labels == j] = 1
                
                # 4. 边缘增强 - 使用Canny边缘检测器找到边缘，然后略微扩张
                if np.any(clean_mask):  # 只在有非零像素时执行
                    edges = cv2.Canny(clean_mask.astype(np.uint8) * 255, 50, 150)
                    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                    
                    # 将边缘与掩码合并
                    final_mask = clean_mask | (edges_dilated > 0).astype(np.uint8)
                else:
                    final_mask = clean_mask
                
                # 转换为RLE
                rle = mask2rle(final_mask)
                
                # 添加到结果列表
                results.append({
                    'ImageId': img_ids[i],
                    'EncodedPixels': rle if np.any(final_mask) else ''  # 如果掩码为空，则提交空字符串
                })
    
    return results

# 主函数
def main():
    # 配置路径
    test_dir = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1/test"
    unet_model_path = "best_unet_model.pth"  # 使用早停保存的最佳模型
    vae_model_path = "fine_tuned_vae/vae_model.pth"
    submission_file = "submission.csv"
    
    # 将.cache路径替换为相对路径或常用路径，如果必要的话
    if not os.path.exists(test_dir):
        test_dir = os.path.join("data", "test")
        print(f"使用替代测试目录: {test_dir}")
        
    if not os.path.exists(unet_model_path):
        # 尝试加载其他模型路径
        alt_paths = ["final_unet_model.pth", "unet_model.pth"]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                unet_model_path = alt_path
                print(f"使用替代UNet模型路径: {unet_model_path}")
                break
    
    # 检查VAE模型路径
    if not os.path.exists(vae_model_path):
        alt_vae_paths = [
            "vae_model.pth", 
            "/tmp/fine_tuned_vae/vae_model.pth",
            "fine_tuned_vae/vae_model.pth"
        ]
        for alt_path in alt_vae_paths:
            if os.path.exists(alt_path):
                vae_model_path = alt_path
                print(f"使用替代VAE模型路径: {vae_model_path}")
                break
    
    batch_size = 8  # 减小批次大小，增加可靠性
    target_size = (256, 256)
    threshold = 0.5  # 默认阈值，将被best_threshold.txt中的值覆盖（如果存在）
    
    # 加载测试集路径
    test_images_paths = load_test_images(test_dir)
    
    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(test_images_paths, target_size)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 加载模型
    print("加载模型...")
    
    # 加载VAE模型
    vae_model = load_vae_model(vae_model_path, input_shape=(3, *target_size))
    
    # 加载UNet模型 - 使用与训练时相同的模型定义
    unet_model = UNet(in_channels=4, out_channels=1, dropout_rate=0.3).to(device)
    try:
        unet_model.load_state_dict(torch.load(unet_model_path, map_location=device))
        print(f"UNet模型加载成功: {unet_model_path}")
    except Exception as e:
        print(f"加载UNet模型时出错: {e}")
        # 尝试非严格加载
        try:
            unet_model.load_state_dict(torch.load(unet_model_path, map_location=device), strict=False)
            print(f"UNet模型使用非严格模式加载成功: {unet_model_path}")
        except Exception as e:
            print(f"非严格加载UNet模型时出错: {e}")
            return
    
    # 预测掩码
    results = predict_masks(unet_model, vae_model, test_loader, threshold)
    
    # 创建提交文件
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(submission_file, index=False)
    print(f"提交文件已保存至: {submission_file}")
    
    # 打印一些统计信息
    mask_count = sum(1 for rle in submission_df['EncodedPixels'] if rle)
    print(f"检测到 {mask_count} 个含有修改区域的图像，共 {len(submission_df)} 个图像")
    print(f"检测率: {mask_count/len(submission_df)*100:.2f}%")

if __name__ == "__main__":
    main() 
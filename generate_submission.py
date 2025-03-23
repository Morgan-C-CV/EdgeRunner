import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

# 从vae_unet_predict.py导入相关函数和模型
from vae_unet_predict import UNet, compute_reconstruction_error

# 尝试导入改进的VAE加载函数
try:
    from resave_vae import load_saved_vae
    has_resave_module = True
    print("成功导入 load_saved_vae 函数")
except ImportError:
    print("警告: 无法导入 load_saved_vae 函数，将使用回退方法")
    has_resave_module = False

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
        except Exception as e:
            print(f"使用新方法加载模型失败: {e}")
            print("回退到传统加载方法...")
    
    # 如果新方法失败，回退到传统方法
    try:
        # 尝试从vae_unet_predict导入VAE模型类
        try:
            from vae_unet_predict import VAE
            # 创建与训练时相同的模型结构
            vae = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
        except ImportError:
            try:
                from sd_vae_finetuning import VAE
                vae = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
            except ImportError:
                print("无法导入VAE类，将使用简化VAE模型...")
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
                vae = SimpleVAE(input_shape=input_shape).to(device)
        
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

# 预测掩码 - 严格遵循unet_validate.py中的实现
def predict_masks(unet_model, vae_model, test_loader, threshold=0.5):
    print("开始预测掩码...")
    results = []
    
    unet_model.eval()
    vae_model.eval()
    
    # 移除加载阈值的代码，使用固定阈值
    print(f"使用固定阈值: {threshold}")
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc="预测中"):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            # 计算重建误差 - 与unet_validate.py一致的实现方式
            try:
                # 尝试使用VAE模型计算重建误差
                recon_error = compute_reconstruction_error(imgs, vae_model)
            except Exception as e:
                print(f"使用VAE计算重建误差时出错: {e}")
                # 如果VAE不工作，使用替代方法生成随机噪声
                _, _, height, width = imgs.shape
                recon_error = torch.rand(batch_size, 1, height, width, device=device) * 0.1
            
            # 组合输入 - 与unet_validate.py一致
            combined_input = torch.cat([imgs, recon_error], dim=1)
            
            # 预测掩码
            mask_preds = unet_model(combined_input)
            
            # 处理每个批次中的图像
            for i in range(batch_size):
                # 获取预测掩码
                mask_np = mask_preds[i].squeeze().cpu().numpy()
                
                # 二值化 - 使用固定阈值，不再根据图像特性自适应调整
                binary_mask = (mask_np > threshold).astype(np.uint8)
                
                # 后处理 - 与unet_validate.py一致
                # 1. 形态学开操作去除噪点
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
                
                # 转换为RLE
                rle = mask2rle(clean_mask)
                
                # 添加到结果列表
                results.append({
                    'ImageId': img_ids[i],
                    'EncodedPixels': rle if np.any(clean_mask) else ''  # 如果掩码为空，则提交空字符串
                })
    
    return results

# 主函数
def main():
    # 配置路径
    test_dir = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1/test/test"
    unet_model_path = "unet_model2.pth" 
    
    # 使用新的VAE模型路径，与unet_validate.py一致
    vae_model_path = "/dev/shm/fine_tuned_vae1/vae_model.pth"
    vae_config_path = "/dev/shm/fine_tuned_vae1/model_config.pth"
    
    submission_file = "submission.csv"
    
    # 处理测试目录路径
    if not os.path.exists(test_dir):
        # 尝试多个可能的测试目录
        alt_test_dirs = [
            os.path.join("data", "test"),
            "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1/test/test",
            "test"
        ]
        for alt_dir in alt_test_dirs:
            if os.path.exists(alt_dir):
                test_dir = alt_dir
                print(f"使用替代测试目录: {test_dir}")
                break
    
    # 处理UNet模型路径 - 与unet_validate.py一致的查找顺序
    if not os.path.exists(unet_model_path):
        alt_paths = ["unet_model.pth", "best_unet_model.pth", "final_unet_model.pth"]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                unet_model_path = alt_path
                print(f"使用替代UNet模型路径: {unet_model_path}")
                break
    
    # 处理VAE模型路径 - 与unet_validate.py一致的查找顺序
    if not os.path.exists(vae_model_path):
        alt_vae_paths = [
            "/dev/shm/fine_tuned_vae/vae_model.pth",
            "/dev/shm/fine_tuned_vae/vae_complete_model.pth",
            "/tmp/fine_tuned_vae/vae_model.pth",
            "fine_tuned_vae/vae_model.pth",
            "vae_model.pth"
        ]
        for alt_path in alt_vae_paths:
            if os.path.exists(alt_path):
                vae_model_path = alt_path
                print(f"使用替代VAE模型路径: {alt_path}")
                break
    
    batch_size = 8
    target_size = (256, 256)
    threshold = 0.8

    test_images_paths = load_test_images(test_dir)

    test_dataset = TestDataset(test_images_paths, target_size)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    print("加载模型...")

    vae_model = load_vae_model(vae_model_path, vae_config_path, input_shape=(3, *target_size))

    unet_model = UNet(in_channels=4, out_channels=1).to(device)
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
    
    # 预测掩码 - 使用固定阈值
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
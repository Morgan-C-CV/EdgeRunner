import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 从之前的VAE模型文件导入必要的类
from sd_vae_finetuning import VAE, ImageDataset, load_dataset_paths

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def load_model(model_path="fine_tuned_vae/vae_model.pth", input_shape=(3, 256, 256), latent_dim=2048):
    """
    加载训练好的VAE模型
    """
    print(f"正在加载模型: {model_path}")
    
    # 创建模型实例
    vae = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
    
    # 加载保存的权重
    if os.path.exists(model_path):
        vae.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功!")
    else:
        print(f"警告: 模型文件 {model_path} 不存在!")
    
    return vae

def apply_post_processing(image, method='none', params=None):
    """
    应用不同的后处理方法减少网格伪影
    
    参数:
        image: numpy数组，形状为(H, W, C)
        method: 后处理方法，可选值：'none', 'gaussian', 'bilateral', 'median', 'nlm', 'fft'
        params: 特定后处理方法的参数
    
    返回:
        处理后的图像
    """
    if method == 'none':
        return image.copy()
    
    elif method == 'gaussian':
        # 高斯模糊，适合轻微网格伪影
        kernel_size = params.get('kernel_size', 3)
        sigma = params.get('sigma', 0.5)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    elif method == 'bilateral':
        # 双边滤波，保留边缘的同时减少网格
        d = params.get('d', 5)
        sigma_color = params.get('sigma_color', 25)
        sigma_space = params.get('sigma_space', 25)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    elif method == 'median':
        # 中值滤波，有效去除椒盐噪声和小网格
        kernel_size = params.get('kernel_size', 3)
        return cv2.medianBlur(image, kernel_size)
    
    elif method == 'nlm':
        # 非局部均值去噪，保留纹理细节
        h = params.get('h', 10)
        template_size = params.get('template_size', 7)
        search_size = params.get('search_size', 21)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_size, search_size)
    
    elif method == 'fft':
        # 基于频域的去网格处理
        # 将图像转为灰度进行FFT
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 执行FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 创建一个高通滤波器来消除网格伪影（网格通常表现为高频规则模式）
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        # 创建掩码
        mask = np.ones((rows, cols), np.uint8)
        
        # 过滤可能的网格频率
        grid_size = params.get('grid_size', 8)  # 估计网格大小
        frequency = rows / grid_size
        bandwidth = params.get('bandwidth', 5)
        
        # 在FFT频域中查找与网格相关的频率并抑制它们
        for i in range(1, 5):  # 处理几个谐波
            freq = int(frequency * i)
            if crow + freq < rows and crow - freq >= 0:
                mask[crow-bandwidth:crow+bandwidth, ccol+freq-bandwidth:ccol+freq+bandwidth] = 0
                mask[crow-bandwidth:crow+bandwidth, ccol-freq-bandwidth:ccol-freq+bandwidth] = 0
            if ccol + freq < cols and ccol - freq >= 0:
                mask[crow+freq-bandwidth:crow+freq+bandwidth, ccol-bandwidth:ccol+bandwidth] = 0
                mask[crow-freq-bandwidth:crow-freq+bandwidth, ccol-bandwidth:ccol+bandwidth] = 0
        
        # 应用掩码并进行反变换
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # 归一化并转回RGB
        img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-8) * 255
        img_back = img_back.astype(np.uint8)
        
        # 只处理亮度通道，保留原始颜色
        result = image.copy()
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = img_back
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    else:
        print(f"未知的后处理方法: {method}")
        return image.copy()

def test_reconstruction_with_post_processing(vae, test_loader, output_dir="enhanced_reconstructions", post_proc_methods=None):
    """
    测试VAE重建，使用多种后处理方法减少网格伪影
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if post_proc_methods is None:
        post_proc_methods = [
            ('none', {}),
            ('gaussian', {'kernel_size': 3, 'sigma': 0.5}),
            ('bilateral', {'d': 5, 'sigma_color': 25, 'sigma_space': 25}),
            ('median', {'kernel_size': 3}),
            ('nlm', {'h': 10, 'template_size': 7, 'search_size': 21}),
            ('fft', {'grid_size': 8, 'bandwidth': 5})
        ]
    
    vae.eval()
    with torch.no_grad():
        for i, imgs in enumerate(tqdm(test_loader, desc="生成重建图像")):
            if i >= 5:  # 只测试5个样本
                break
                
            imgs = imgs.to(device)
            reconstructions = vae(imgs)
            
            # 转换回numpy格式并从[-1,1]恢复到[0,255]
            imgs_np = ((imgs.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).astype(np.uint8)
            recon_np = ((reconstructions.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).astype(np.uint8)
            
            for j in range(imgs_np.shape[0]):
                original_img = imgs_np[j]
                recon_img = recon_np[j].copy()
                
                # 保存单独的处理结果
                for method, params in post_proc_methods:
                    processed_img = apply_post_processing(recon_img, method, params)
                    cv2.imwrite(
                        os.path.join(output_dir, f"recon_{i}_{j}_{method}.png"),
                        cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                    )
                
                # 创建水平对比图（OpenCV格式）
                all_images = [original_img]
                for method, params in post_proc_methods:
                    processed_img = apply_post_processing(recon_img, method, params)
                    all_images.append(processed_img)
                
                # 确保所有图像具有相同的尺寸
                image_height = original_img.shape[0]
                image_width = original_img.shape[1]
                
                # 创建标题图像
                title_height = 30
                title_image = np.ones((title_height, image_width * len(all_images), 3), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # 添加标题
                titles = ["Original"] + [method for method, _ in post_proc_methods]
                for k, title in enumerate(titles):
                    cv2.putText(
                        title_image, 
                        title, 
                        (k * image_width + 10, title_height - 10), 
                        font, 0.5, (0, 0, 0), 1
                    )
                
                # 水平拼接所有图像
                comparison = np.hstack(all_images)
                
                # 垂直拼接标题和图像
                result = np.vstack([title_image, comparison])
                
                # 保存最终对比图
                cv2.imwrite(
                    os.path.join(output_dir, f"all_methods_{i}_{j}.png"),
                    cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                )
    
    print(f"重建结果已保存到 {output_dir}")
    print("各种后处理方法说明:")
    print("- none: 无后处理")
    print("- gaussian: 高斯模糊 - 轻微平滑，可减少小网格")
    print("- bilateral: 双边滤波 - 保留边缘细节的同时减少网格")
    print("- median: 中值滤波 - 移除离群噪点和细小网格")
    print("- nlm: 非局部均值去噪 - 保留纹理细节的同时减少噪声")
    print("- fft: 基于频域的去网格 - 特别针对规则网格图案")

def analyze_latent_space(vae, test_loader, num_samples=5):
    """
    分析潜在空间特性，可能与网格伪影有关
    """
    vae.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for i, imgs in enumerate(test_loader):
            if i >= num_samples:
                break
            
            imgs = imgs.to(device)
            latent = vae.encoder(imgs)
            latent_vectors.append(latent.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # 分析潜在向量的统计特性
    mean = np.mean(latent_vectors, axis=0)
    std = np.std(latent_vectors, axis=0)
    min_val = np.min(latent_vectors, axis=0)
    max_val = np.max(latent_vectors, axis=0)
    
    print("潜在空间分析:")
    print(f"形状: {latent_vectors.shape}")
    print(f"均值: {np.mean(mean):.4f}")
    print(f"标准差: {np.mean(std):.4f}")
    print(f"最小值: {np.mean(min_val):.4f}")
    print(f"最大值: {np.mean(max_val):.4f}")
    
    # 检查是否有规则模式，这可能与网格伪影有关
    freq_analysis = np.fft.fft(latent_vectors, axis=1)
    power = np.abs(freq_analysis)**2
    dominant_freq = np.argmax(power[:, 1:], axis=1) + 1
    
    print(f"主导频率分布: {np.bincount(dominant_freq)}")
    
    return latent_vectors

def main():
    # 定义参数
    model_path = "fine_tuned_vae/vae_model.pth"  # 模型路径
    input_shape = (3, 256, 256)  # 输入图像尺寸
    latent_dim = 2048  # 潜在空间维度
    batch_size = 4  # 批次大小
    test_images_path = None  # 测试图像路径，如果为None则使用训练集的一部分
    
    # 加载模型
    vae = load_model(model_path, input_shape, latent_dim)
    
    # 加载测试数据
    if test_images_path:
        image_paths = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        # 使用训练数据作为测试
        dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"
        image_paths = load_dataset_paths(dataset_path, max_samples=20)  # 只用少量样本
    
    # 创建测试数据集和数据加载器
    test_dataset = ImageDataset(image_paths, target_size=input_shape[1:])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义要测试的后处理方法
    post_proc_methods = [
        ('none', {}),  # 无处理
        ('gaussian', {'kernel_size': 3, 'sigma': 0.5}),  # 轻微高斯模糊
        ('bilateral', {'d': 5, 'sigma_color': 25, 'sigma_space': 25}),  # 双边滤波
        ('gaussian', {'kernel_size': 5, 'sigma': 1.0}),  # 较强高斯模糊
        ('median', {'kernel_size': 3}),  # 中值滤波
        ('fft', {'grid_size': 8, 'bandwidth': 3})  # 频域去网格
    ]
    
    # 测试重建效果并应用各种后处理方法
    test_reconstruction_with_post_processing(vae, test_loader, post_proc_methods=post_proc_methods)
    
    # 分析潜在空间特性
    analyze_latent_space(vae, test_loader)

if __name__ == "__main__":
    main() 
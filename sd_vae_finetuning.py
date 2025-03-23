import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义VAE模型
class Encoder(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), latent_dim=2048):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            
            nn.Flatten()
        )
        
        # 计算展平后的特征维度
        self.feature_size = 512 * (input_shape[1] // 16) * (input_shape[2] // 16)
        
        # 添加一个线性层将特征映射到潜在空间
        self.fc = nn.Linear(self.feature_size, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim=2048, output_shape=(3, 256, 256)):
        super(ImprovedDecoder, self).__init__()
        
        self.initial_height = output_shape[1] // 16
        self.initial_width = output_shape[2] // 16
        
        self.fc = nn.Linear(latent_dim, 256 * self.initial_height * self.initial_width)
        
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 添加残差块
        self.res1 = ResBlock(256)
        self.res2 = ResBlock(128)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 这里插入 res2
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        # 在解码器中添加更多细节层
        self.final_detail = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResBlock(32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.initial_height, self.initial_width)
        x = self.initial_block(x)
        x = self.res1(x)
        x = self.decoder[0:2](x)  # 前两层
        x = self.res2(x)
        x = self.decoder[2:](x)   # 剩余层
        return x

class VAE(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), latent_dim=4*64*64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = ImprovedDecoder(latent_dim, input_shape)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_pretrained_weights(vae, weights_path=None):
    if weights_path and os.path.exists(weights_path):
        print(f"加载预训练权重: {weights_path}")
        vae.load_state_dict(torch.load(weights_path))
    return vae

class ImageDataset(Dataset):
    def __init__(self, image_paths, target_size=(256, 256)):
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            img = cv2.imread(img_path)
            if img is None:
                # 如果图像无法读取，返回一个随机噪声图像
                img = np.random.randn(self.target_size[0], self.target_size[1], 3) * 0.1 + 0.5
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为PyTorch张量并归一化到[-1, 1]
            img_tensor = self.transform(img)
            
            return img_tensor
        except Exception as e:
            print(f"处理图像 {self.image_paths[idx]} 出错: {e}")
            # 返回随机噪声作为替代
            return torch.randn(3, self.target_size[0], self.target_size[1])

def load_dataset_paths(dataset_path, max_samples=None):
    print("加载数据集路径...")
    originals = []
    
    for dirname, _, filenames in os.walk(os.path.join(dataset_path, 'train/train')):
        for filename in filenames:
            if "originals" in dirname:
                originals.append(os.path.join(dirname, filename))
                if max_samples is not None and len(originals) >= max_samples:
                    break
    
    print(f"加载了 {len(originals)} 个原始图像路径")
    return originals

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        
    def forward(self, input, target):
        # 转换为灰度
        input_gray = 0.299 * input[:, 0:1] + 0.587 * input[:, 1:2] + 0.114 * input[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # 计算梯度
        input_grad_x = nn.functional.conv2d(input_gray, self.sobel_x, padding=1)
        input_grad_y = nn.functional.conv2d(input_gray, self.sobel_y, padding=1)
        target_grad_x = nn.functional.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = nn.functional.conv2d(target_gray, self.sobel_y, padding=1)
        
        # 添加epsilon防止数值不稳定
        epsilon = 1e-6
        input_edge = torch.sqrt(input_grad_x**2 + input_grad_y**2 + epsilon)
        target_edge = torch.sqrt(target_grad_x**2 + target_grad_y**2 + epsilon)
        
        # 返回边缘差异的 L1 损失
        return nn.functional.l1_loss(input_edge, target_edge)

def finetune_vae(vae, train_loader, epochs=10):
    vae.train()
    optimizer = optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-5)  # 降低学习率和权重衰减
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    pixel_criterion = nn.L1Loss()
    edge_criterion = EdgeLoss()
    
    # 添加梯度值监控和记录
    grad_stats = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        pixel_running_loss = 0.0
        edge_running_loss = 0.0
        
        for i, imgs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            outputs = vae(imgs)
            
            # 分别计算两种损失
            pixel_loss = pixel_criterion(outputs, imgs)
            edge_loss = edge_criterion(outputs, imgs)
            
            # 检查损失是否为NaN，如果是则跳过此批次
            if torch.isnan(pixel_loss) or torch.isnan(edge_loss):
                print(f"检测到NaN损失! pixel_loss: {pixel_loss.item()}, edge_loss: {edge_loss.item()}")
                continue
                
            # 使用较小的权重组合损失
            loss = pixel_loss + 0.05 * edge_loss  # 降低边缘损失权重
            
            loss.backward()
            
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            # 监控梯度
            if i % 50 == 0:
                max_grad = 0.0
                for param in vae.parameters():
                    if param.grad is not None:
                        param_max = param.grad.abs().max().item()
                        max_grad = max(max_grad, param_max)
                grad_stats.append(max_grad)
                print(f"最大梯度值: {max_grad:.6f}")
            
            optimizer.step()
            
            running_loss += loss.item()
            pixel_running_loss += pixel_loss.item()
            edge_running_loss += edge_loss.item()
            
            if i % 50 == 49:
                print(f"[{epoch + 1}, {i + 1}] Total Loss: {running_loss / 50:.4f}, Pixel Loss: {pixel_running_loss / 50:.4f}, Edge Loss: {edge_running_loss / 50:.4f}")
                running_loss = 0.0
                pixel_running_loss = 0.0
                edge_running_loss = 0.0
        
        # 计算整个epoch的平均损失并调整学习率
        epoch_loss = running_loss / max(1, len(train_loader))  # 防止除零
        scheduler.step()
        print(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}")
    
    return vae, grad_stats

def test_reconstruction(vae, test_loader, output_dir="reconstructions"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    vae.eval()
    with torch.no_grad():
        for i, imgs in enumerate(test_loader):
            if i >= 10:  # 只测试10个样本
                break
                
            imgs = imgs.to(device)
            reconstructions = vae(imgs)
            
            # 转换回numpy格式并从[-1,1]恢复到[0,255]
            imgs_np = ((imgs.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).astype(np.uint8)
            recon_np = ((reconstructions.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5).astype(np.uint8)
            
            for j in range(imgs_np.shape[0]):
                # 增强对比度
                recon_enhanced = cv2.convertScaleAbs(recon_np[j], alpha=1.1, beta=5)
                comparison = np.hstack([imgs_np[j], recon_enhanced])
                cv2.imwrite(os.path.join(output_dir, f"reconstruction_{i*imgs_np.shape[0]+j}.png"), 
                           cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print(f"保存了重建样本到 {output_dir}")

def save_model(vae, output_dir="fine_tuned_vae"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    torch.save(vae.state_dict(), os.path.join(output_dir, "/dev/shm/fine_tuned_vae/vae_model.pth"))
    torch.save(vae.encoder.state_dict(), os.path.join(output_dir, "encoder_model.pth"))
    torch.save(vae.decoder.state_dict(), os.path.join(output_dir, "decoder_model.pth"))
    
    print(f"模型已保存到 {output_dir}")

# 使用VGG特征提取器计算感知损失
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 暂时移除VGG16，避免磁盘空间不足问题
        # vgg = torchvision.models.vgg16(pretrained=True)
        # self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
            
    def forward(self, input, target):
        # 暂时直接返回0，不使用感知损失
        return torch.tensor(0.0, device=input.device)
        # input_features = self.feature_extractor(input)
        # target_features = self.feature_extractor(target)
        # return nn.functional.mse_loss(input_features, target_features)

def main():
    dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"
    
    batch_size = 16
    epochs = 35
    target_size = (256, 256)
    max_samples = None  # 取消样本数量限制
    latent_dim = 2048  # 增加潜在空间维度
    
    image_paths = load_dataset_paths(dataset_path, max_samples)
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(image_paths, target_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 创建测试数据加载器（使用少量样本）
    test_dataset = ImageDataset(image_paths[:10], target_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建并配置模型
    vae = VAE(input_shape=(3, *target_size), latent_dim=latent_dim).to(device)
    
    # 可选：加载预训练权重
    # vae = load_pretrained_weights(vae, "pretrained_weights.pth")
    
    # 冻结编码器，只训练解码器
    for i, (name, param) in enumerate(vae.encoder.named_parameters()):
        if i > len(list(vae.encoder.parameters())) // 2:  # 解冻后半部分
            param.requires_grad = True
    for param in vae.decoder.parameters():
        param.requires_grad = True
    
    # 打印模型结构
    print(vae)
    
    # 训练模型 - 不使用感知损失
    vae, grad_stats = finetune_vae(vae, train_loader, epochs)
    
    # 保存模型
    save_model(vae)
    
    # 测试重建
    test_reconstruction(vae, test_loader)

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random

# 导入基本模型
from sd_vae_finetuning import VAE, Encoder, ImprovedDecoder, ResBlock
# 导入新的加载函数
from resave_vae import load_saved_vae

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置随机种子以提高可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

set_seed(42)

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


# 加载VAE模型
def load_vae_model(model_path, config_path=None, input_shape=(3, 256, 256), latent_dim=2048):
    print(f"尝试从 {model_path} 加载VAE模型...")
    
    # 首先尝试使用新的加载方法
    if os.path.exists(model_path):
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


# 数据集类
class CombinedDataset(Dataset):
    def __init__(self, originals_paths, images_paths, masks_paths, target_size=(256, 256)):
        self.originals_paths = originals_paths
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.originals_paths)

    def __getitem__(self, idx):
        try:
            # 加载原始图像、修改图像和掩码
            orig_path = self.originals_paths[idx]
            mod_path = self.images_paths[idx]
            mask_path = self.masks_paths[idx]

            orig = cv2.imread(orig_path)
            mod = cv2.imread(mod_path)
            mask = cv2.imread(mask_path, 0)

            if orig is None or mod is None or mask is None:
                raise ValueError(f"无法读取图像: {orig_path}, {mod_path}, {mask_path}")

            # 调整大小
            orig = cv2.resize(orig, self.target_size, interpolation=cv2.INTER_LINEAR)
            mod = cv2.resize(mod, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            # 转换颜色空间
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            mod = cv2.cvtColor(mod, cv2.COLOR_BGR2RGB)

            # 转换为张量
            orig_tensor = self.transform(orig)
            mod_tensor = self.transform(mod)
            mask = (mask > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            return orig_tensor, mod_tensor, mask_tensor

        except Exception as e:
            print(f"处理图像 {self.originals_paths[idx]} 时出错: {e}")
            # 返回随机张量作为替代
            return (torch.randn(3, *self.target_size),
                    torch.randn(3, *self.target_size),
                    torch.zeros(1, *self.target_size))


# 计算重建误差 - 针对可能的模型结构不一致问题进行容错处理
def compute_reconstruction_error(orig_image, vae_model):
    with torch.no_grad():
        try:
            # 尝试正常重建
            recon = vae_model(orig_image)
            # 计算像素级绝对差异，在通道维度上求平均
            error = torch.mean(torch.abs(orig_image - recon), dim=1, keepdim=True)
        except Exception as e:
            print(f"重建过程发生错误，使用随机噪声作为重建误差: {e}")
            # 发生错误时，使用随机噪声作为替代
            batch_size, _, height, width = orig_image.shape
            error = torch.rand(batch_size, 1, height, width, device=orig_image.device) * 0.1
    return error


# 训练U-Net模型
def train_unet(unet_model, vae_model, train_loader, val_loader=None, epochs=10):
    optimizer = optim.Adam(unet_model.parameters(), lr=1e-4)

    # 损失函数: 结合BCE和Dice损失
    def dice_loss(y_pred, y_true, smooth=1):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    # BCE + Dice损失
    def bce_dice_loss(y_pred, y_true):
        bce = nn.BCELoss()(y_pred, y_true)
        dice = dice_loss(y_pred, y_true)
        return bce + dice

    for epoch in range(epochs):
        unet_model.train()
        running_loss = 0.0

        for i, (orig_imgs, mod_imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            orig_imgs = orig_imgs.to(device)
            mod_imgs = mod_imgs.to(device)
            masks = masks.to(device)

            # 计算重建误差
            recon_error = compute_reconstruction_error(orig_imgs, vae_model)

            # 组合修改图像和重建误差
            combined_input = torch.cat([mod_imgs, recon_error], dim=1)

            # 前向传播
            optimizer.zero_grad()
            outputs = unet_model(combined_input)

            # 计算损失
            loss = bce_dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # 验证
        if val_loader:
            unet_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for orig_imgs, mod_imgs, masks in val_loader:
                    orig_imgs = orig_imgs.to(device)
                    mod_imgs = mod_imgs.to(device)
                    masks = masks.to(device)

                    recon_error = compute_reconstruction_error(orig_imgs, vae_model)
                    combined_input = torch.cat([mod_imgs, recon_error], dim=1)

                    outputs = unet_model(combined_input)
                    loss = bce_dice_loss(outputs, masks)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1} 验证损失: {val_loss:.4f}")

    return unet_model


# 预测函数
def predict_mask(image_path, vae_model, unet_model, target_size=(256, 256)):
    # 加载和预处理图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 计算重建误差
    with torch.no_grad():
        # 使用VAE重建
        recon = vae_model(img_tensor)

        # 计算重建误差
        error = torch.mean(torch.abs(img_tensor - recon), dim=1, keepdim=True)

        # 组合输入
        combined_input = torch.cat([img_tensor, error], dim=1)

        # 预测掩码
        mask_pred = unet_model(combined_input)

    # 后处理
    mask_np = mask_pred.squeeze().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

    # 可选: 应用形态学操作来改善掩码质量
    kernel = np.ones((5, 5), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    mask_np = cv2.erode(mask_np, kernel, iterations=1)

    return mask_np


# 加载数据集路径
def load_dataset_paths(dataset_path, max_samples=None):
    print("加载数据集路径...")
    originals = []
    images = []
    masks = []

    for dirname, _, filenames in os.walk(os.path.join(dataset_path, 'train/train')):
        for filename in filenames:
            if "originals" in dirname:
                originals.append(os.path.join(dirname, filename))
            elif "images" in dirname:
                images.append(os.path.join(dirname, filename))
            elif "masks" in dirname:
                masks.append(os.path.join(dirname, filename))

            # 如果达到最大样本数，停止加载
            if max_samples is not None and len(originals) >= max_samples:
                break

    print(f"加载了 {len(originals)} 个原始图像, {len(images)} 个修改图像, {len(masks)} 个掩码")

    # 确保三个列表长度一致
    min_len = min(len(originals), len(images), len(masks))
    return originals[:min_len], images[:min_len], masks[:min_len]


# 增强的数据集类 - 添加更多数据增强
class EnhancedCombinedDataset(Dataset):
    def __init__(self, originals_paths, images_paths, masks_paths, target_size=(256, 256), augment=True):
        self.originals_paths = originals_paths
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.target_size = target_size
        self.augment = augment
        
        # 基本变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.originals_paths)

    def __getitem__(self, idx):
        try:
            # 加载原始图像、修改图像和掩码
            orig_path = self.originals_paths[idx]
            mod_path = self.images_paths[idx]
            mask_path = self.masks_paths[idx]

            orig = cv2.imread(orig_path)
            mod = cv2.imread(mod_path)
            mask = cv2.imread(mask_path, 0)

            if orig is None or mod is None or mask is None:
                raise ValueError(f"无法读取图像: {orig_path}, {mod_path}, {mask_path}")

            # 调整大小
            orig = cv2.resize(orig, self.target_size, interpolation=cv2.INTER_LINEAR)
            mod = cv2.resize(mod, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # 数据增强
            if self.augment:
                # 随机水平翻转
                if random.random() > 0.5:
                    orig = cv2.flip(orig, 1)
                    mod = cv2.flip(mod, 1)
                    mask = cv2.flip(mask, 1)
                
                # 随机垂直翻转
                if random.random() > 0.5:
                    orig = cv2.flip(orig, 0)
                    mod = cv2.flip(mod, 0)
                    mask = cv2.flip(mask, 0)
                
                # 随机旋转
                if random.random() > 0.5:
                    angle = random.randint(-15, 15)
                    height, width = orig.shape[:2]
                    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                    
                    orig = cv2.warpAffine(orig, matrix, (width, height), flags=cv2.INTER_LINEAR)
                    mod = cv2.warpAffine(mod, matrix, (width, height), flags=cv2.INTER_LINEAR)
                    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
                
                # 随机亮度和对比度变化
                if random.random() > 0.5:
                    alpha = 1.0 + random.uniform(-0.2, 0.2)  # 对比度
                    beta = random.uniform(-10, 10)  # 亮度
                    
                    orig = cv2.convertScaleAbs(orig, alpha=alpha, beta=beta)
                    mod = cv2.convertScaleAbs(mod, alpha=alpha, beta=beta)

            # 转换颜色空间
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            mod = cv2.cvtColor(mod, cv2.COLOR_BGR2RGB)

            # 转换为张量
            orig_tensor = self.transform(orig)
            mod_tensor = self.transform(mod)
            mask = (mask > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            return orig_tensor, mod_tensor, mask_tensor

        except Exception as e:
            print(f"处理图像 {self.originals_paths[idx]} 时出错: {e}")
            # 返回随机张量作为替代
            return (torch.randn(3, *self.target_size),
                    torch.randn(3, *self.target_size),
                    torch.zeros(1, *self.target_size))


# 增强的重建误差计算 - 多尺度特征融合
def enhanced_reconstruction_error(orig_image, vae_model):
    with torch.no_grad():
        try:
            # 正常重建
            recon = vae_model(orig_image)
            
            # 基本误差 - 像素级绝对差异
            pixel_error = torch.abs(orig_image - recon)
            
            # 计算多个尺度的误差
            # 1. 原始尺度
            error_scale1 = torch.mean(pixel_error, dim=1, keepdim=True)
            
            # 2. 降采样尺度 - 捕获更大的结构差异
            orig_down = nn.functional.avg_pool2d(orig_image, kernel_size=2)
            recon_down = nn.functional.avg_pool2d(recon, kernel_size=2)
            error_scale2 = torch.mean(torch.abs(orig_down - recon_down), dim=1, keepdim=True)
            error_scale2 = nn.functional.interpolate(error_scale2, size=orig_image.shape[2:], mode='bilinear')
            
            # 3. 边缘检测差异 - 修正Sobel算子的应用方式
            # 定义水平和垂直的Sobel核
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(device)
            
            # 扩展到3D卷积核
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            # 分别对RGB三个通道应用Sobel算子
            edge_diff = torch.zeros_like(error_scale1)
            for c in range(3):
                # 提取单通道
                orig_channel = orig_image[:, c:c+1, :, :]
                recon_channel = recon[:, c:c+1, :, :]
                
                # 分别计算原始和重建图像的梯度
                orig_grad_x = nn.functional.conv2d(orig_channel, sobel_x, padding=1)
                orig_grad_y = nn.functional.conv2d(orig_channel, sobel_y, padding=1)
                
                recon_grad_x = nn.functional.conv2d(recon_channel, sobel_x, padding=1)
                recon_grad_y = nn.functional.conv2d(recon_channel, sobel_y, padding=1)
                
                # 计算梯度差异并累加
                channel_edge_diff_x = torch.abs(orig_grad_x - recon_grad_x)
                channel_edge_diff_y = torch.abs(orig_grad_y - recon_grad_y)
                channel_edge_diff = torch.sqrt(channel_edge_diff_x**2 + channel_edge_diff_y**2)
                
                edge_diff = edge_diff + channel_edge_diff / 3.0
            
            # 加权融合不同尺度的误差
            combined_error = 0.5 * error_scale1 + 0.3 * error_scale2 + 0.2 * edge_diff
            
            return combined_error
        except Exception as e:
            print(f"增强重建误差计算出错: {e}")
            # 发生错误时，回退到基本版本
            return compute_reconstruction_error(orig_image, vae_model)


# 改进的损失函数 - 添加Focal Loss和边缘感知损失
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, edge_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # BCE和Dice损失的权重因子
        self.gamma = gamma  # Focal Loss的gamma参数
        self.edge_weight = edge_weight  # 边缘损失的权重
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, pred, target):
        # 基本的BCE损失
        bce_loss = self.bce(pred, target)
        
        # Focal Loss分量 - 处理类别不平衡
        pt = target * pred + (1 - target) * (1 - pred)  # = p_t
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        focal_loss = focal_loss.mean()
        
        # Dice Loss分量
        smooth = 1e-6
        intersection = torch.sum(target * pred)
        dice_coeff = (2. * intersection + smooth) / (torch.sum(target) + torch.sum(pred) + smooth)
        dice_loss = 1 - dice_coeff
        
        # 边缘感知损失 - 使用Sobel算子 - 修复实现方式
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(target.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(target.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # 计算目标掩码的梯度
        target_grad_x = nn.functional.conv2d(target, sobel_x, padding=1)
        target_grad_y = nn.functional.conv2d(target, sobel_y, padding=1)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)
        
        # 计算预测掩码的梯度
        pred_grad_x = nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = nn.functional.conv2d(pred, sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
        
        # 边缘损失 - 边缘上的差异应该更少
        edge_loss = nn.functional.mse_loss(pred_grad, target_grad)
        
        # 组合所有损失
        total_loss = self.alpha * focal_loss + (1 - self.alpha) * dice_loss + self.edge_weight * edge_loss
        
        return total_loss


# 阈值寻优函数
def find_optimal_threshold(model, val_loader, vae_model, device, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    best_f1 = 0
    best_threshold = 0.5
    
    model.eval()
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for orig_imgs, mod_imgs, masks in tqdm(val_loader, desc="寻找最佳阈值"):
            orig_imgs = orig_imgs.to(device)
            mod_imgs = mod_imgs.to(device)
            masks = masks.to(device)
            
            # 使用增强的重建误差
            recon_error = enhanced_reconstruction_error(orig_imgs, vae_model)
            combined_input = torch.cat([mod_imgs, recon_error], dim=1)
            
            outputs = model(combined_input)
            
            all_preds.append(outputs.cpu())
            all_masks.append(masks.cpu())
    
    # 合并所有批次
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 测试不同阈值
    for threshold in thresholds:
        binary_preds = (all_preds > threshold).float()
        
        # 计算F1分数
        smooth = 1e-6
        true_pos = torch.sum(binary_preds * all_masks)
        false_pos = torch.sum(binary_preds * (1 - all_masks))
        false_neg = torch.sum((1 - binary_preds) * all_masks)
        
        precision = true_pos / (true_pos + false_pos + smooth)
        recall = true_pos / (true_pos + false_neg + smooth)
        
        f1 = 2 * (precision * recall) / (precision + recall + smooth)
        
        print(f"阈值: {threshold:.2f}, F1分数: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")
    
    # 保存最佳阈值
    with open("best_threshold.txt", "w") as f:
        f.write(str(best_threshold))
    
    return best_threshold


# 学习率调度器
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr_factor = max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return lr_factor
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 改进的训练函数，集成前面的优化
def train_improved_unet(model, vae_model, train_loader, val_loader=None, epochs=35, patience=7):
    # 用Adam优化器替代，增加权重衰减(L2正则化)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # 使用余弦退火学习率
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10%的warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # 使用改进的损失函数
    criterion = CombinedLoss(alpha=0.5, gamma=2.0, edge_weight=0.2)
    
    # 早停策略
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 训练循环
        for i, (orig_imgs, mod_imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            orig_imgs = orig_imgs.to(device)
            mod_imgs = mod_imgs.to(device)
            masks = masks.to(device)
            
            # 使用增强的重建误差
            recon_error = enhanced_reconstruction_error(orig_imgs, vae_model)
            
            # 组合修改图像和重建误差
            combined_input = torch.cat([mod_imgs, recon_error], dim=1)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(combined_input)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                print(f"Epoch {epoch + 1}, Batch {i + 1}: loss = {running_loss / 10:.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")
                train_losses.append(running_loss / 10)
                running_loss = 0.0
        
        # 验证
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for orig_imgs, mod_imgs, masks in val_loader:
                    orig_imgs = orig_imgs.to(device)
                    mod_imgs = mod_imgs.to(device)
                    masks = masks.to(device)
                    
                    recon_error = enhanced_reconstruction_error(orig_imgs, vae_model)
                    combined_input = torch.cat([mod_imgs, recon_error], dim=1)
                    
                    outputs = model(combined_input)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1} 验证损失: {val_loss:.4f}")
            
            # 检查是否需要早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve_epochs = 0
                
                # 每次改进都保存模型
                torch.save(model.state_dict(), f"unet_model_epoch_{epoch+1}.pth")
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= patience:
                print(f"早停! {patience} 个epoch没有改进")
                # 恢复最佳模型
                model.load_state_dict(best_model_state)
                break
        
        # 每5个epoch执行一次阈值优化
        if val_loader and (epoch + 1) % 5 == 0:
            best_threshold = find_optimal_threshold(model, val_loader, vae_model, device)
            print(f"Epoch {epoch + 1} 最佳阈值: {best_threshold:.2f}")
    
    # 训练结束，如果有最佳模型状态，则加载它
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 保存训练历史
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    
    return model


# 主函数 - 使用改进的模型和训练流程
def main():
    # 配置参数
    dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"
    
    # 使用新的VAE模型路径
    vae_model_path = "/dev/shm/fine_tuned_vae1/vae_model.pth"
    vae_config_path = "/dev/shm/fine_tuned_vae1/model_config.pth"
    
    batch_size = 8
    epochs = 35
    target_size = (256, 256)
    max_samples = None

    # 加载数据集路径
    originals, images, masks = load_dataset_paths(dataset_path, max_samples)

    # 划分训练集和验证集
    split_idx = int(len(originals) * 0.9)  # 90%用于训练，10%用于验证
    train_originals, val_originals = originals[:split_idx], originals[split_idx:]
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]

    # 创建增强的数据集
    train_dataset = EnhancedCombinedDataset(train_originals, train_images, train_masks, target_size, augment=True)
    val_dataset = EnhancedCombinedDataset(val_originals, val_images, val_masks, target_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载VAE模型（使用新的带配置的加载方式）
    vae_model = load_vae_model(vae_model_path, vae_config_path, input_shape=(3, *target_size))

    # 创建改进的UNet模型
    improved_unet = ImprovedUNet(in_channels=4, out_channels=1, dropout_rate=0.3).to(device)
    
    # 使用改进的训练函数
    improved_unet = train_improved_unet(
        improved_unet, 
        vae_model, 
        train_loader, 
        val_loader, 
        epochs=epochs,
        patience=7
    )

    # 保存最终模型
    torch.save(improved_unet.state_dict(), "improved_unet_model.pth")
    print("改进的UNet模型已保存")
    
    # 执行最终的阈值优化
    best_threshold = find_optimal_threshold(
        improved_unet, 
        val_loader, 
        vae_model, 
        device,
        thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    )
    print(f"最终最佳阈值: {best_threshold:.2f}")
    
    # 预测并保存一些测试样例
    improved_unet.eval()
    for i in range(10):
        if i < len(images):
            test_image_path = images[i]
            img = cv2.imread(test_image_path)
            if img is None:
                print(f"无法读取图像: {test_image_path}")
                continue
            
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 计算增强的重建误差
            with torch.no_grad():
                recon_error = enhanced_reconstruction_error(img_tensor, vae_model)
                combined_input = torch.cat([img_tensor, recon_error], dim=1)
                mask_pred = improved_unet(combined_input)
            
            # 使用最佳阈值
            mask_np = mask_pred.squeeze().cpu().numpy()
            mask_np = (mask_np > best_threshold).astype(np.uint8) * 255
            
            # 形态学后处理
            kernel = np.ones((3, 3), np.uint8)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 保存结果
            cv2.imwrite(f"improved_mask_{i}.png", mask_np)

if __name__ == "__main__":
    main()
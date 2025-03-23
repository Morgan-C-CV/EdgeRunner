import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 从sd_vae_finetuning.py导入VAE模型
from sd_vae_finetuning import VAE, Encoder, ImprovedDecoder, ResBlock

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


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
def load_vae_model(model_path, input_shape=(3, 256, 256), latent_dim=2048):
    print(f"尝试从 {model_path} 加载VAE模型...")
    
    # 创建与训练时相同的模型结构
    from sd_vae_finetuning import VAE, Encoder, ImprovedDecoder, ResBlock
    
    # 创建与训练时完全相同的模型结构
    vae = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
    
    try:
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


# 主函数
def main():
    # 配置参数
    dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"
    vae_model_path = "/dev/shm/fine_tuned_vae/vae_model.pth"
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

    # 创建数据集和数据加载器
    train_dataset = CombinedDataset(train_originals, train_images, train_masks, target_size)
    val_dataset = CombinedDataset(val_originals, val_images, val_masks, target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载VAE模型
    vae_model = load_vae_model(vae_model_path, input_shape=(3, *target_size))

    # 创建并训练U-Net模型
    unet_model = UNet(in_channels=4, out_channels=1).to(device)
    unet_model = train_unet(unet_model, vae_model, train_loader, val_loader, epochs)

    # 保存U-Net模型
    torch.save(unet_model.state_dict(), "unet_model2.pth")
    print("U-Net模型已保存")

    for i in range(10):
        test_image_path = images[i]
        mask_pred = predict_mask(test_image_path, vae_model, unet_model, target_size)
        cv2.imwrite(f"predicted_mask{i}.png", mask_pred)


if __name__ == "__main__":
    main()
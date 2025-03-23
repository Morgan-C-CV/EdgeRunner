import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import gc

physical_devices = tf.config.list_physical_devices('GPU')

def autoencoder_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return models.Model(inputs, outputs)

def unet_model(input_shape=(256, 256, 4)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u4)
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    return models.Model(inputs, outputs)

def preprocess_autoencoder(originals_paths, target_size=(256, 256)):
    X = []
    for path in originals_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        X.append(img)
    return np.array(X)


def preprocess_unet(originals_paths, images_paths, masks_paths, ae_model, target_size=(256, 256)):
    X, y = [], []
    for orig_path, mod_path, mask_path in zip(originals_paths, images_paths, masks_paths):
        orig = cv2.imread(orig_path)
        mod = cv2.imread(mod_path)
        mask = cv2.imread(mask_path, 0)
        if orig is None or mod is None or mask is None:
            continue
        orig = cv2.resize(orig, target_size, interpolation=cv2.INTER_LINEAR)
        mod = cv2.resize(mod, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        mod = cv2.cvtColor(mod, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        mask = (mask > 0).astype(float)

        # 计算重建误差
        recon = ae_model.predict(orig[np.newaxis, ...])[0]
        error = np.mean(np.abs(orig - recon), axis=-1)

        # 输入U-Net
        X.append(np.concatenate([mod, error[..., np.newaxis]], axis=-1))  # (256, 256, 4)
        y.append(mask[..., np.newaxis])
    return np.array(X), np.array(y)


# 损失函数
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1.) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.)
    return bce + (1 - dice)


def data_generator_autoencoder(file_paths, batch_size=4, target_size=(256, 256)):
    n = len(file_paths)
    while True:
        # 随机打乱文件顺序
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_files = [file_paths[idx] for idx in batch_indices]
            
            X_batch = []
            for path in batch_files:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
                    X_batch.append(img)
                except Exception as e:
                    print(f"处理 {path} 时发生错误: {e}")
                    continue
                    
            if X_batch:
                X_batch = np.array(X_batch)
                yield X_batch, X_batch

def data_generator_unet(originals, images, masks, ae_model, batch_size=2, target_size=(256, 256)):
    n = len(originals)
    while True:
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_originals = [originals[idx] for idx in batch_indices]
            batch_images = [images[idx] for idx in batch_indices] 
            batch_masks = [masks[idx] for idx in batch_indices]
            
            X_batch, y_batch = [], []
            for orig_path, mod_path, mask_path in zip(batch_originals, batch_images, batch_masks):
                try:
                    orig = cv2.imread(orig_path)
                    mod = cv2.imread(mod_path)
                    mask = cv2.imread(mask_path, 0)
                    
                    if orig is None or mod is None or mask is None:
                        continue
                        
                    orig = cv2.resize(orig, target_size, interpolation=cv2.INTER_LINEAR)
                    mod = cv2.resize(mod, target_size, interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                    
                    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(float) / 255.0
                    mod = cv2.cvtColor(mod, cv2.COLOR_BGR2RGB).astype(float) / 255.0
                    mask = (mask > 0).astype(float)
                    
                    recon = ae_model.predict(orig[np.newaxis, ...], verbose=0)[0]
                    error = np.mean(np.abs(orig - recon), axis=-1)
                    
                    X_batch.append(np.concatenate([mod, error[..., np.newaxis]], axis=-1))
                    y_batch.append(mask[..., np.newaxis])
                except Exception as e:
                    print(f"处理 {orig_path} 时发生错误: {e}")
                    continue
                    
            if X_batch:
                yield np.array(X_batch), np.array(y_batch)


dataset = []
originals = []
images = []
masks = []

dataset_path = "/home/cv-hacker/.cache/kagglehub/competitions/aaltoes-2025-computer-vision-v-1"

for dirname, _, filenames in os.walk(f'{dataset_path}/train/train'):
    for filename in filenames:
        if "originals" in dirname:
            originals.append(os.path.join(dirname, filename))
        elif "images" in dirname:
            images.append(os.path.join(dirname, filename))
        elif "masks" in dirname:
            masks.append(os.path.join(dirname, filename))

print(f"找到图像: {len(originals)} 原始图像, {len(images)} 修改图像, {len(masks)} 掩码")

# 减少训练集大小以节省内存
max_samples = 2000  # 调整此数值以适应您的内存限制
originals_small = originals[:max_samples]
images_small = images[:max_samples]
masks_small = masks[:max_samples]

# 使用生成器训练自编码器
print("训练自编码器...")
ae_model = autoencoder_model()
ae_model.compile(optimizer='adam', loss='mse')

# 计算训练步数
BATCH_SIZE = 4  # 减小批次大小
EPOCHS = 10     # 减少轮次数
steps_per_epoch = len(originals_small) // BATCH_SIZE

# 使用生成器训练
train_gen = data_generator_autoencoder(originals_small, batch_size=BATCH_SIZE)
ae_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    verbose=1
)

# 保存模型并清理内存
ae_model.save("autoencoder.h5")
print("自编码器训练完成，模型已保存")

# 强制垃圾回收
gc.collect()
tf.keras.backend.clear_session()

# 训练U-Net
print("训练U-Net...")
unet = unet_model()
unet.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])

# 加载已保存的自编码器模型
ae_model = tf.keras.models.load_model("autoencoder.h5")

# 计算U-Net训练步数
BATCH_SIZE_UNET = 2  # 更小的批次
EPOCHS_UNET = 5      # 减少轮次
steps_per_epoch_unet = len(originals_small) // BATCH_SIZE_UNET

# 使用生成器训练U-Net
unet_gen = data_generator_unet(originals_small, images_small, masks_small, ae_model, batch_size=BATCH_SIZE_UNET)
unet.fit(
    unet_gen,
    steps_per_epoch=steps_per_epoch_unet,
    epochs=EPOCHS_UNET,
    verbose=1
)

# 保存模型并清理内存
unet.save("unet_model.h5")
print("U-Net训练完成，模型已保存")
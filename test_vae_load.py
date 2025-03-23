import os
import torch
import torch.nn as nn
from sd_vae_finetuning import VAE, Encoder, ImprovedDecoder, ResBlock

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def test_load_vae(model_path):
    """测试加载VAE模型"""
    print(f"尝试从 {model_path} 加载VAE模型...")
    
    # 创建模型
    input_shape = (3, 256, 256)
    latent_dim = 2048
    vae = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
    
    try:
        # 尝试加载状态字典
        vae.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 成功加载状态字典")
        
        # 测试模型
        with torch.no_grad():
            vae.eval()
            test_input = torch.randn(1, 3, 256, 256).to(device)
            test_output = vae(test_input)
            print(f"✅ 模型推理成功: 输入形状 {test_input.shape}, 输出形状 {test_output.shape}")
            return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

def test_load_complete_vae(model_path):
    """测试加载完整VAE模型"""
    print(f"尝试从 {model_path} 加载完整VAE模型...")
    
    try:
        # 尝试加载完整模型
        vae = torch.load(model_path, map_location=device)
        print("✅ 成功加载完整模型")
        
        # 测试模型
        with torch.no_grad():
            vae.eval()
            test_input = torch.randn(1, 3, 256, 256).to(device)
            test_output = vae(test_input)
            print(f"✅ 模型推理成功: 输入形状 {test_input.shape}, 输出形状 {test_output.shape}")
            return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

if __name__ == "__main__":
    # 定义原始路径和新路径
    original_path = "/dev/shm/fine_tuned_vae/vae_model.pth"
    new_path = "/dev/shm/fine_tuned_vae1/vae_model.pth"
    new_complete_path = "/dev/shm/fine_tuned_vae1/vae_complete_model.pth"
    
    print("\n=== 测试原始模型加载 ===")
    orig_success = test_load_vae(original_path)
    
    print("\n=== 测试新目录中的模型加载 ===")
    new_success = test_load_vae(new_path)
    
    print("\n=== 测试新目录中的完整模型加载 ===")
    complete_success = test_load_complete_vae(new_complete_path)
    
    print("\n=== 测试结果摘要 ===")
    print(f"原始模型加载: {'成功' if orig_success else '失败'}")
    print(f"新目录模型加载: {'成功' if new_success else '失败'}")
    print(f"新目录完整模型加载: {'成功' if complete_success else '失败'}")
    
    if new_success or complete_success:
        print("\n✅ 修复成功! 您现在可以使用新目录中的模型了:")
        print(f"   状态字典: {new_path}")
        print(f"   完整模型: {new_complete_path}")
    else:
        print("\n❌ 修复失败，请检查错误消息并重试") 
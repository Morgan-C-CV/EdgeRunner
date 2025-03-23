import os
import torch
import torch.nn as nn
import copy
from sd_vae_finetuning import VAE, Encoder, ImprovedDecoder, ResBlock

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def analyze_state_dict(state_dict):
    """分析状态字典的结构和形状"""
    print("\n=== 状态字典分析 ===")
    
    # 按模块整理参数
    modules = {}
    for k, v in state_dict.items():
        parts = k.split('.')
        if len(parts) > 1:
            module = parts[0]
            if module not in modules:
                modules[module] = []
            modules[module].append((k, v.shape))
    
    # 打印每个模块的参数
    for module, params in modules.items():
        print(f"\n模块: {module} (共 {len(params)} 个参数)")
        for name, shape in sorted(params):
            print(f"  {name}: {shape}")
    
    return modules

def manual_copy_tensors(old_state_dict, new_model):
    """手动将旧状态字典中的张量复制到新模型中，尽可能匹配形状"""
    print("\n=== 尝试手动复制张量 ===")
    
    new_state_dict = new_model.state_dict()
    matched_count = 0
    mismatch_count = 0
    missing_count = 0
    
    # 创建一个新的状态字典
    updated_state_dict = {}
    
    # 尝试直接复制相同键名和形状的张量
    for new_key, new_param in new_state_dict.items():
        if new_key in old_state_dict and old_state_dict[new_key].shape == new_param.shape:
            # 形状完全匹配
            updated_state_dict[new_key] = old_state_dict[new_key]
            matched_count += 1
        else:
            # 检查是否可以部分匹配
            matched = False
            
            # 特殊处理解码器部分
            if "decoder" in new_key:
                # 如果是卷积或全连接层权重，尝试匹配其它层
                for old_key, old_param in old_state_dict.items():
                    if "decoder" in old_key and old_param.dim() == new_param.dim():
                        # 尝试找到对应的层
                        if ("weight" in new_key and "weight" in old_key) or ("bias" in new_key and "bias" in old_key):
                            try:
                                # 尝试将形状不一样的张量调整到新形状
                                reshaped_param = resize_tensor(old_param, new_param.shape)
                                if reshaped_param is not None:
                                    updated_state_dict[new_key] = reshaped_param
                                    print(f"调整形状: {old_key} {old_param.shape} -> {new_key} {new_param.shape}")
                                    matched = True
                                    break
                            except Exception as e:
                                continue
            
            if not matched:
                print(f"无法匹配: {new_key} {new_param.shape}")
                # 保持随机初始化
                missing_count += 1
    
    print(f"匹配的参数: {matched_count}")
    print(f"未匹配的参数: {missing_count}")
    
    # 加载匹配的参数
    new_model.load_state_dict(updated_state_dict, strict=False)
    return new_model

def resize_tensor(tensor, new_shape):
    """尝试将张量调整为新形状，如果可能的话"""
    if tensor.dim() != len(new_shape):
        return None
    
    # 创建目标张量
    result = torch.zeros(new_shape, device=tensor.device)
    
    # 对于每个维度，尝试复制尽可能多的元素
    if tensor.dim() == 1:  # 偏置项
        min_size = min(tensor.size(0), new_shape[0])
        result[:min_size] = tensor[:min_size]
    elif tensor.dim() == 2:  # 全连接层
        min_size0 = min(tensor.size(0), new_shape[0])
        min_size1 = min(tensor.size(1), new_shape[1])
        result[:min_size0, :min_size1] = tensor[:min_size0, :min_size1]
    elif tensor.dim() == 4:  # 卷积层
        min_size0 = min(tensor.size(0), new_shape[0])
        min_size1 = min(tensor.size(1), new_shape[1])
        min_size2 = min(tensor.size(2), new_shape[2])
        min_size3 = min(tensor.size(3), new_shape[3])
        result[:min_size0, :min_size1, :min_size2, :min_size3] = tensor[:min_size0, :min_size1, :min_size2, :min_size3]
    
    return result

def create_matching_decoder(original_state_dict):
    """基于原始状态字典创建匹配的解码器结构"""
    
    # 分析原始state_dict中的decoder结构
    decoder_keys = [k for k in original_state_dict.keys() if k.startswith('decoder.')]
    
    # 创建一个与原始匹配的解码器
    class MatchingDecoder(nn.Module):
        def __init__(self, latent_dim=2048, output_shape=(3, 256, 256)):
            super(MatchingDecoder, self).__init__()
            
            self.initial_height = output_shape[1] // 16
            self.initial_width = output_shape[2] // 16
            
            # 从原始状态字典获取形状信息
            fc_weight_shape = original_state_dict.get('decoder.fc.weight', torch.zeros(256 * 16 * 16, latent_dim)).shape
            fc_out_features = fc_weight_shape[0]
            
            self.fc = nn.Linear(latent_dim, fc_out_features)
            
            # 检查原始字典中是否有initial_block
            if any('decoder.initial_block' in k for k in decoder_keys):
                self.initial_block = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
            
            # 检查是否有残差块
            if any('decoder.res1' in k for k in decoder_keys):
                self.res1 = ResBlock(256)
            
            if any('decoder.res2' in k for k in decoder_keys):
                self.res2 = ResBlock(128)
            
            # 检查是否有detail_enhance
            if any('decoder.detail_enhance' in k for k in decoder_keys):
                # 获取形状信息
                de_0_weight_shape = original_state_dict.get('decoder.detail_enhance.0.weight', torch.zeros(16, 3, 3, 3)).shape
                de_2_weight_shape = original_state_dict.get('decoder.detail_enhance.2.weight', torch.zeros(16, 16, 3, 3)).shape
                de_4_weight_shape = original_state_dict.get('decoder.detail_enhance.4.weight', torch.zeros(3, 16, 3, 3)).shape
                
                self.detail_enhance = nn.Sequential(
                    nn.Conv2d(de_0_weight_shape[1], de_0_weight_shape[0], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(de_2_weight_shape[1], de_2_weight_shape[0], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(de_4_weight_shape[1], de_4_weight_shape[0], kernel_size=3, padding=1),
                    nn.ReLU()
                )
            
            # 解码器各层
            has_decoder = any('decoder.decoder' in k for k in decoder_keys)
            if has_decoder:
                self.decoder = nn.Sequential(
                    nn.Sequential(
                        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 3, kernel_size=3, padding=1),
                        nn.Tanh()
                    )
                )
            
            # 检查是否有smoothing
            if any('decoder.smoothing' in k for k in decoder_keys):
                smooth_weight_shape = original_state_dict.get('decoder.smoothing.0.weight', torch.zeros(3, 1, 3, 3)).shape
                self.smoothing = nn.Sequential(
                    nn.Conv2d(smooth_weight_shape[1]*3, smooth_weight_shape[0], kernel_size=3, padding=1, groups=3)
                )
        
        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, 256, self.initial_height, self.initial_width)
            
            if hasattr(self, 'initial_block'):
                x = self.initial_block(x)
            
            if hasattr(self, 'res1'):
                x = self.res1(x)
            
            # 解码器路径
            if hasattr(self, 'decoder'):
                x = self.decoder[0](x)
                
                if hasattr(self, 'res2'):
                    x = self.res2(x)
                
                x = self.decoder[1](x)
                x = self.decoder[2](x)
                x = self.decoder[3](x)
            
            # 应用细节增强（如果存在）
            if hasattr(self, 'detail_enhance'):
                x = self.detail_enhance(x)
            
            # 应用平滑处理（如果存在）
            if hasattr(self, 'smoothing'):
                x = self.smoothing(x)
            
            return x
    
    return MatchingDecoder

# 将CustomVAE类移到全局作用域
class CustomVAE(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), latent_dim=2048, matching_decoder_class=None):
        super(CustomVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_shape, latent_dim)
        
        if matching_decoder_class is not None:
            self.decoder = matching_decoder_class(latent_dim, input_shape)
        else:
            # 使用默认的ImprovedDecoder作为回退选项
            self.decoder = ImprovedDecoder(latent_dim, input_shape)
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def resave_vae_model():
    """加载现有VAE模型，使用新的结构重新保存到新目录"""
    
    # 加载源模型
    source_path = "/dev/shm/fine_tuned_vae/vae_model.pth"
    
    if not os.path.exists(source_path):
        print(f"源模型路径不存在: {source_path}")
        return False
    
    print(f"从 {source_path} 加载源模型...")
    
    # 加载源模型状态字典
    state_dict = torch.load(source_path, map_location=device)
    
    # 分析原始状态字典
    analyze_state_dict(state_dict)
    
    # 创建自定义解码器
    MatchingDecoder = create_matching_decoder(state_dict)
    
    # 创建自定义的VAE（现在使用全局定义的CustomVAE类）
    input_shape = (3, 256, 256)
    latent_dim = 2048
    new_vae = CustomVAE(input_shape=input_shape, latent_dim=latent_dim, 
                         matching_decoder_class=MatchingDecoder).to(device)
    
    # 打印源模型键和目标模型键
    source_keys = set(state_dict.keys())
    target_keys = set(new_vae.state_dict().keys())
    
    print(f"源模型参数数量: {len(source_keys)}")
    print(f"目标模型参数数量: {len(target_keys)}")
    
    # 尝试加载模型
    try:
        print("尝试加载状态字典...")
        new_vae.load_state_dict(state_dict, strict=False)
        print("使用非严格模式成功加载模型")
    except Exception as e:
        print(f"状态字典加载失败: {e}")
        print("尝试手动复制参数...")
        new_vae = manual_copy_tensors(state_dict, new_vae)
    
    # 确保新目录存在
    new_dir = "/dev/shm/fine_tuned_vae1/"
    os.makedirs(new_dir, exist_ok=True)
    
    # 保存到新路径
    output_path = os.path.join(new_dir, "vae_model.pth")
    torch.save(new_vae.state_dict(), output_path)
    print(f"成功保存模型状态字典到 {output_path}")
    
    # 保存模型结构信息到文本文件
    model_structure_path = os.path.join(new_dir, "model_structure.txt")
    with open(model_structure_path, 'w') as f:
        f.write(str(new_vae))
    print(f"模型结构已保存到 {model_structure_path}")
    
    # 保存模型的配置信息（便于重新构建）
    config_path = os.path.join(new_dir, "model_config.pth")
    config = {
        'input_shape': input_shape,
        'latent_dim': latent_dim,
        'has_initial_block': hasattr(new_vae.decoder, 'initial_block'),
        'has_res1': hasattr(new_vae.decoder, 'res1'),
        'has_res2': hasattr(new_vae.decoder, 'res2'),
        'has_detail_enhance': hasattr(new_vae.decoder, 'detail_enhance'),
        'has_smoothing': hasattr(new_vae.decoder, 'smoothing'),
    }
    torch.save(config, config_path)
    print(f"模型配置已保存到 {config_path}")
    
    # 验证模型是否可用
    try:
        with torch.no_grad():
            test_input = torch.randn(1, 3, 256, 256).to(device)
            test_output = new_vae(test_input)
            print(f"模型验证成功: 输入形状 {test_input.shape}, 输出形状 {test_output.shape}")
            return True
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False

# 记录原始状态字典在解码器部分的所有键
def print_decoder_keys(state_dict_path):
    state_dict = torch.load(state_dict_path, map_location=device)
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
    print("\n解码器部分的键:")
    for k in sorted(decoder_keys):
        print(f"  {k}")

def load_saved_vae(state_dict_path, config_path=None):
    """
    加载重保存的VAE模型，自动匹配解码器结构
    
    Args:
        state_dict_path: 状态字典路径
        config_path: 可选的配置文件路径，用于构建匹配的解码器
        
    Returns:
        加载了权重的VAE模型
    """
    print(f"从 {state_dict_path} 加载状态字典...")
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # 尝试加载配置
    config = None
    if config_path and os.path.exists(config_path):
        try:
            config = torch.load(config_path, map_location=device)
            print(f"从 {config_path} 加载模型配置")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    # 分析状态字典以确定解码器结构
    decoder_type = "custom"
    input_shape = (3, 256, 256)
    latent_dim = 2048
    
    if config:
        input_shape = config.get('input_shape', input_shape)
        latent_dim = config.get('latent_dim', latent_dim)
    
    # 基于状态字典创建匹配的解码器
    MatchingDecoder = create_matching_decoder(state_dict)
    
    # 创建自定义VAE
    vae = CustomVAE(
        input_shape=input_shape, 
        latent_dim=latent_dim,
        matching_decoder_class=MatchingDecoder
    ).to(device)
    
    # 加载状态字典
    try:
        vae.load_state_dict(state_dict, strict=False)
        print("成功加载模型状态字典")
    except Exception as e:
        print(f"加载状态字典时出错: {e}")
        print("尝试手动匹配参数...")
        vae = manual_copy_tensors(state_dict, vae)
    
    return vae

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("/dev/shm/fine_tuned_vae1/", exist_ok=True)
    
    # 先打印原始状态字典的解码器键
    print_decoder_keys("/dev/shm/fine_tuned_vae/vae_model.pth")
    
    # 重新保存模型
    success = resave_vae_model()
    print(f"模型重新保存{'成功' if success else '失败'}")
    
    if success:
        # 测试加载我们的自定义模型
        try:
            print("\n=== 测试加载自定义模型 ===")
            custom_vae = load_saved_vae(
                "/dev/shm/fine_tuned_vae1/vae_model.pth",
                "/dev/shm/fine_tuned_vae1/model_config.pth"
            )
            custom_vae.eval()
            
            with torch.no_grad():
                test_input = torch.randn(1, 3, 256, 256).to(device)
                test_output = custom_vae(test_input)
                print(f"自定义模型测试成功: 输出形状 {test_output.shape}")
                
            # 生成示例代码，帮助用户在其他脚本中加载模型
            example_code = """
# 在其他脚本中使用此模型的示例代码:
from resave_vae import load_saved_vae

# 加载自定义模型
model = load_saved_vae(
    "/dev/shm/fine_tuned_vae1/vae_model.pth",
    "/dev/shm/fine_tuned_vae1/model_config.pth"
)

# 使用模型进行推理
model.eval()
import torch
with torch.no_grad():
    # 自定义图像预处理...
    test_input = torch.randn(1, 3, 256, 256).to(model.device)
    output = model(test_input)
"""
            example_path = os.path.join("/dev/shm/fine_tuned_vae1/", "usage_example.py")
            with open(example_path, 'w') as f:
                f.write(example_code)
            print(f"使用示例已保存到 {example_path}")
                
        except Exception as e:
            print(f"自定义模型测试失败: {e}")
            
        # 为了保持兼容性，尝试使用原始VAE类加载模型（这会失败，因为形状不匹配）
        try:
            print("\n=== 尝试使用原始VAE类加载（仅供参考） ===")
            test_vae = VAE(input_shape=(3, 256, 256), latent_dim=2048).to(device)
            test_vae.load_state_dict(torch.load("/dev/shm/fine_tuned_vae1/vae_model.pth", map_location=device), strict=False)
            test_vae.eval()
            
            with torch.no_grad():
                test_input = torch.randn(1, 3, 256, 256).to(device)
                test_output = test_vae(test_input)
                print(f"原始模型结构测试成功: 输出形状 {test_output.shape}")
        except Exception as e:
            print(f"原始模型结构测试失败（预期的）: {str(e)[:200]}...")
            print("请使用load_saved_vae函数加载此模型") 
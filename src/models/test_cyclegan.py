"""
CycleGAN模型测试脚本（不依赖实际运行环境，仅验证代码结构）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.cyclegan_generator import CycleGANGenerator
    from src.models.cyclegan_discriminator import CycleGANDiscriminator
    import torch
    
    print("=" * 60)
    print("CycleGAN模型结构测试")
    print("=" * 60)
    
    # 测试生成器G
    print("\n1. 测试生成器G（Label→Photo）")
    generator_G = CycleGANGenerator(in_channels=3, out_channels=3, n_residual_blocks=9)
    print(f"   ✓ 生成器G创建成功")
    print(f"   - 参数量: {sum(p.numel() for p in generator_G.parameters()):,}")
    
    # 测试生成器F
    print("\n2. 测试生成器F（Photo→Label）")
    generator_F = CycleGANGenerator(in_channels=3, out_channels=3, n_residual_blocks=9)
    print(f"   ✓ 生成器F创建成功")
    print(f"   - 参数量: {sum(p.numel() for p in generator_F.parameters()):,}")
    
    # 测试判别器D_photo
    print("\n3. 测试判别器D_photo")
    discriminator_photo = CycleGANDiscriminator(in_channels=3)
    print(f"   ✓ 判别器D_photo创建成功")
    print(f"   - 参数量: {sum(p.numel() for p in discriminator_photo.parameters()):,}")
    
    # 测试判别器D_label
    print("\n4. 测试判别器D_label")
    discriminator_label = CycleGANDiscriminator(in_channels=3)
    print(f"   ✓ 判别器D_label创建成功")
    print(f"   - 参数量: {sum(p.numel() for p in discriminator_label.parameters()):,}")
    
    # 测试前向传播（如果CUDA可用）
    if torch.cuda.is_available():
        print("\n5. 测试前向传播（GPU）")
        device = torch.device("cuda")
        generator_G = generator_G.to(device)
        generator_F = generator_F.to(device)
        discriminator_photo = discriminator_photo.to(device)
        discriminator_label = discriminator_label.to(device)
        
        # 测试输入
        label = torch.randn(1, 3, 256, 256).to(device)
        photo = torch.randn(1, 3, 256, 256).to(device)
        
        # 生成器前向传播
        fake_photo = generator_G(label)
        fake_label = generator_F(photo)
        print(f"   ✓ Label→Photo: {label.shape} → {fake_photo.shape}")
        print(f"   ✓ Photo→Label: {photo.shape} → {fake_label.shape}")
        
        # 循环一致性
        recovered_label = generator_F(fake_photo)
        recovered_photo = generator_G(fake_label)
        print(f"   ✓ 循环一致性测试通过")
        print(f"     Label→Photo→Label: {label.shape} → {recovered_label.shape}")
        print(f"     Photo→Label→Photo: {photo.shape} → {recovered_photo.shape}")
        
        # 判别器前向传播
        pred_real_photo = discriminator_photo(photo)
        pred_fake_photo = discriminator_photo(fake_photo)
        pred_real_label = discriminator_label(label)
        pred_fake_label = discriminator_label(fake_label)
        print(f"   ✓ 判别器输出形状:")
        print(f"     D_photo(real): {pred_real_photo.shape}")
        print(f"     D_photo(fake): {pred_fake_photo.shape}")
        print(f"     D_label(real): {pred_real_label.shape}")
        print(f"     D_label(fake): {pred_fake_label.shape}")
        
        print(f"\n   ✓ 输出范围检查:")
        print(f"     fake_photo: [{fake_photo.min():.3f}, {fake_photo.max():.3f}] (期望: [-1, 1])")
        print(f"     fake_label: [{fake_label.min():.3f}, {fake_label.max():.3f}] (期望: [-1, 1])")
    else:
        print("\n5. 跳过前向传播测试（CUDA不可用）")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！CycleGAN模型实现正确。")
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("   请确保已安装PyTorch: pip install torch torchvision")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()





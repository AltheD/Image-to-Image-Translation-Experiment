"""
CycleGAN生成器（ResNet-based架构）。

架构：
- 编码器：3个下采样卷积层
- 残差块：6-9个ResNet块（论文推荐9个）
- 解码器：3个上采样转置卷积层
- 使用Instance Normalization替代Batch Normalization
- 输出：Tanh激活，范围[-1, 1]

两个生成器：
- G: Label → Photo
- F: Photo → Label
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """ResNet残差块（使用Instance Normalization）"""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(in_channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """
    CycleGAN生成器（ResNet-based）。
    
    输入：图像 [B, 3, 256, 256]
    输出：生成的图像 [B, 3, 256, 256]，范围[-1, 1]（Tanh激活）
    
    Args:
        in_channels: 输入通道数（默认3）
        out_channels: 输出通道数（默认3）
        n_residual_blocks: 残差块数量（默认9，论文推荐）
    """
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        
        # 编码器：3个下采样卷积层
        # 第一层：C7s1-64（7×7卷积，stride=1，输出64通道）
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 下采样1：C3s2-128（3×3卷积，stride=2，输出128通道）
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 下采样2：C3s2-256（3×3卷积，stride=2，输出256通道）
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 残差块：n_residual_blocks个ResNet块
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # 解码器：3个上采样转置卷积层
        self.decoder = nn.Sequential(
            # 上采样1：C3s2-128（转置卷积，stride=2，输出128通道）
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 上采样2：C3s2-64（转置卷积，stride=2，输出64通道）
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 最后一层：C7s1-3（7×7卷积，stride=1，输出3通道，Tanh激活）
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=0, bias=True),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # 编码
        x = self.encoder(x)
        # 残差块
        x = self.residual_blocks(x)
        # 解码
        x = self.decoder(x)
        return x


def test_cyclegan_generator():
    """测试CycleGAN生成器的前向传播"""
    # 测试生成器G（Label→Photo）
    generator_G = CycleGANGenerator(in_channels=3, out_channels=3, n_residual_blocks=9)
    generator_G.eval()
    
    # 测试输入
    x = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = generator_G(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Expected range: [-1, 1] (Tanh activation)")
    
    # 计算参数量
    total_params = sum(p.numel() for p in generator_G.parameters())
    trainable_params = sum(p.numel() for p in generator_G.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试生成器F（Photo→Label）
    generator_F = CycleGANGenerator(in_channels=3, out_channels=3, n_residual_blocks=9)
    generator_F.eval()
    
    with torch.no_grad():
        output_F = generator_F(x)
        print(f"\nGenerator F output shape: {output_F.shape}")
        print(f"Generator F output range: [{output_F.min():.3f}, {output_F.max():.3f}]")
    
    print("\n✅ CycleGAN generator test passed!")


if __name__ == "__main__":
    test_cyclegan_generator()





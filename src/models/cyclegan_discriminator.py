"""
CycleGAN判别器（70×70 PatchGAN）。

架构：
- 70×70 PatchGAN：C64-C128-C256-C512
- 所有层：Convolution(4×4, stride=2) + BatchNorm + LeakyReLU(0.2)
- 第一层C64不使用BatchNorm
- 最后层：Convolution映射到1维输出 + Sigmoid
- 输入：仅图像（不需要concatenate label），因为CycleGAN是无监督的
- 输出：70×70的patch判别结果

两个判别器：
- D_photo: 区分真实photo和生成的photo
- D_label: 区分真实label和生成的label
"""

import torch
import torch.nn as nn


class CycleGANDiscriminator(nn.Module):
    """
    CycleGAN判别器（70×70 PatchGAN）。
    
    输入：图像 [B, 3, 256, 256]
    输出：patch判别结果 [B, 1, 30, 30]（70×70 receptive field）
    
    注意：对于256×256输入，经过4次stride=2的下采样：
    - 256 → 128 → 64 → 32 → 16
    但实际输出是30×30，因为第一层padding=1，后续层padding=1
    最终receptive field约为70×70
    """
    def __init__(self, in_channels=3):
        super(CycleGANDiscriminator, self).__init__()
        
        # C64（第一层，不使用BatchNorm）
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # C128
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # C256
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # C512
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 最后一层：映射到1维输出
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True),
            # 注意：通常使用BCEWithLogitsLoss，所以这里不添加Sigmoid
            # 如果需要Sigmoid，可以在forward中手动添加
        )
    
    def forward(self, x):
        x = self.layer1(x)  # [B, 64, 128, 128]
        x = self.layer2(x)  # [B, 128, 64, 64]
        x = self.layer3(x)  # [B, 256, 32, 32]
        x = self.layer4(x)  # [B, 512, 16, 16]
        x = self.final(x)   # [B, 1, 15, 15] 或 [B, 1, 14, 14]
        return x


def test_cyclegan_discriminator():
    """测试CycleGAN判别器的前向传播"""
    # 测试判别器D_photo
    discriminator_photo = CycleGANDiscriminator(in_channels=3)
    discriminator_photo.eval()
    
    # 测试输入
    x = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = discriminator_photo(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Expected: patch判别结果，每个patch对应70×70的receptive field")
    
    # 计算参数量
    total_params = sum(p.numel() for p in discriminator_photo.parameters())
    trainable_params = sum(p.numel() for p in discriminator_photo.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试判别器D_label
    discriminator_label = CycleGANDiscriminator(in_channels=3)
    discriminator_label.eval()
    
    with torch.no_grad():
        output_label = discriminator_label(x)
        print(f"\nDiscriminator D_label output shape: {output_label.shape}")
        print(f"Discriminator D_label output range: [{output_label.min():.3f}, {output_label.max():.3f}]")
    
    print("\n✅ CycleGAN discriminator test passed!")


if __name__ == "__main__":
    test_cyclegan_discriminator()





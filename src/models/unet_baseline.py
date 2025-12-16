"""
U-Net基线模型（仅L1损失，无GAN）。

架构：与Pix2Pix生成器相同的U-Net结构
- 编码器：C64-C128-C256-C512-C512-C512-C512-C512
- 解码器：CD512-CD512-CD512-C512-C256-C128-C64
- Skip Connections：编码器第i层连接到解码器第n-i层
"""

import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    """U-Net的基础块"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.down = down
        
        if down:
            # 编码器：Convolution(4×4, stride=2) + BatchNorm + LeakyReLU(0.2)
            # 第一层不使用BatchNorm
            if in_channels == 3:  # 第一层
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
                self.norm = None
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
                self.norm = nn.BatchNorm2d(out_channels)
            self.activation = nn.LeakyReLU(0.2, inplace=True)
            # 为接口一致性补上 dropout 属性（仅解码器使用）
            self.dropout = None
        else:
            # 解码器：Convolution + BatchNorm + Dropout(0.5) + ReLU
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout(0.5) if use_dropout else None
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class UNetBaseline(nn.Module):
    """
    U-Net基线模型（与Pix2Pix生成器相同的架构）。
    
    输入：label图像 [B, 3, 256, 256]
    输出：生成的photo图像 [B, 3, 256, 256]，范围[-1, 1]（Tanh激活）
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetBaseline, self).__init__()
        
        # 编码器（下采样）
        # C64 (第一层，无BatchNorm)
        self.down1 = UNetBlock(in_channels, 64, down=True)
        # C128
        self.down2 = UNetBlock(64, 128, down=True)
        # C256
        self.down3 = UNetBlock(128, 256, down=True)
        # C512
        self.down4 = UNetBlock(256, 512, down=True)
        # C512
        self.down5 = UNetBlock(512, 512, down=True)
        # C512
        self.down6 = UNetBlock(512, 512, down=True)
        # C512
        self.down7 = UNetBlock(512, 512, down=True)
        # C512 (最底层，无下采样，直接卷积)
        # 注意：此层输出空间尺寸为 1×1，BatchNorm/InstanceNorm 在训练模式下会因为
        # “每通道/每实例只有 1 个元素” 报错，因此这里不做归一化，只使用 Conv + 激活。
        self.down8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 解码器（上采样，带skip connections）
        # CD512 (带Dropout)
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        # CD512 (带Dropout)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)  # 512 + 512 (skip)
        # CD512 (带Dropout)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)  # 512 + 512 (skip)
        # C512 (无Dropout)
        self.up4 = UNetBlock(1024, 512, down=False, use_dropout=False)  # 512 + 512 (skip)
        # C256
        self.up5 = UNetBlock(1024, 256, down=False, use_dropout=False)  # 512 + 512 (skip)
        # C128
        self.up6 = UNetBlock(512, 128, down=False, use_dropout=False)  # 256 + 256 (skip)
        # C64
        self.up7 = UNetBlock(256, 64, down=False, use_dropout=False)  # 128 + 128 (skip)
        
        # 最后一层：映射到3通道 + Tanh
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1, bias=True),  # 64 + 64 (skip)
            nn.Tanh()
        )
    
    def forward(self, x):
        # 编码器（保存中间特征用于skip connections）
        d1 = self.down1(x)      # [B, 64, 128, 128]
        d2 = self.down2(d1)     # [B, 128, 64, 64]
        d3 = self.down3(d2)     # [B, 256, 32, 32]
        d4 = self.down4(d3)     # [B, 512, 16, 16]
        d5 = self.down5(d4)     # [B, 512, 8, 8]
        d6 = self.down6(d5)     # [B, 512, 4, 4]
        d7 = self.down7(d6)     # [B, 512, 2, 2]
        d8 = self.down8(d7)     # [B, 512, 1, 1]
        
        # 解码器（带skip connections）
        u1 = self.up1(d8)       # [B, 512, 2, 2]
        u1 = torch.cat([u1, d7], dim=1)  # [B, 1024, 2, 2]
        
        u2 = self.up2(u1)       # [B, 512, 4, 4]
        u2 = torch.cat([u2, d6], dim=1)  # [B, 1024, 4, 4]
        
        u3 = self.up3(u2)       # [B, 512, 8, 8]
        u3 = torch.cat([u3, d5], dim=1)  # [B, 1024, 8, 8]
        
        u4 = self.up4(u3)       # [B, 512, 16, 16]
        u4 = torch.cat([u4, d4], dim=1)  # [B, 1024, 16, 16]
        
        u5 = self.up5(u4)       # [B, 256, 32, 32]
        u5 = torch.cat([u5, d3], dim=1)  # [B, 512, 32, 32]
        
        u6 = self.up6(u5)       # [B, 128, 64, 64]
        u6 = torch.cat([u6, d2], dim=1)  # [B, 256, 64, 64]
        
        u7 = self.up7(u6)       # [B, 64, 128, 128]
        u7 = torch.cat([u7, d1], dim=1)  # [B, 128, 128, 128]
        
        # 最后一层
        output = self.final(u7)  # [B, 3, 256, 256]
        
        return output


def test_unet_baseline():
    """测试U-Net基线模型的前向传播"""
    model = UNetBaseline(in_channels=3, out_channels=3)
    model.eval()
    
    # 测试输入
    x = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Expected range: [-1, 1] (Tanh activation)")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ U-Net baseline model test passed!")


if __name__ == "__main__":
    test_unet_baseline()


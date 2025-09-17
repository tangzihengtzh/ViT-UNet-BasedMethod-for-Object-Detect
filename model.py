import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import vit_tiny_patch16_224, VisionTransformer


class ViTEncoderWithSkips(nn.Module):
    """ViT encoder that returns multi‑level token features for skip connections."""

    def __init__(self, pretrained: bool = True, return_layers=(3, 6, 9)):
        super().__init__()
        self.vit: VisionTransformer = vit_tiny_patch16_224(pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim  # 192 for vit‑tiny
        self.grid_size = self.vit.patch_embed.grid_size  # (14, 14)
        self.return_layers = set(return_layers)

    def forward(self, x):
        B = x.size(0)
        x = self.vit.patch_embed(x)  # [B, N, D]
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        skips = []
        for idx, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if idx in self.return_layers:
                skips.append(x[:, 1:, :])
                # print(x[:, 1:, :].shape)

        x = self.vit.norm(x)
        skips.append(x[:, 1:, :])
        return skips  # len= len(return_layers)+1


class UpBlock(nn.Module):
    """上采样 + 拼接 skip + 双卷积"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # print()
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:  #
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SkipDecoder(nn.Module):
    """U‑Net style decoder consuming ViT token features."""

    def __init__(self, embed_dim: int, out_channels: int = 4):
        super().__init__()
        # 通道规划（深→浅）
        c1, c2, c3, c4 = embed_dim, embed_dim // 2, embed_dim // 4, embed_dim // 8

        # Token→特征图映射，全部先降到统一 embed_dim，再各自投影
        self.proj = nn.ModuleList([
            nn.Conv2d(embed_dim, c4, 1),  #
            nn.Conv2d(embed_dim, c3, 1),
            nn.Conv2d(embed_dim, c2, 1),
            nn.Conv2d(embed_dim, c1, 1),  #
        ])

        # 上采样模块（深→浅）
        self.up3 = UpBlock(c1, c2, c2)  # 14 → 28
        self.up2 = UpBlock(c2, c3, c3)  # 28 → 56
        self.up1 = UpBlock(c3, c4, c4)  # 56 → 112

        self.final_up = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)  # 112 → 224
        self.head = nn.Conv2d(c4, out_channels, kernel_size=1)

    def forward(self, feats, grid_size):
        """feats: [skip1, skip2, skip3, deep] (token), grid_size=(Hfeat, Wfeat)"""
        B = feats[0].size(0)
        Hf, Wf = grid_size

        # token → feature map
        fmap = []
        for idx, feat in enumerate(feats):
            f = feat.transpose(1, 2).reshape(B, -1, Hf, Wf)  # [B, D, 14, 14]
            fmap.append(self.proj[idx](f))

        # 浅→深: fmap[0]=最浅(c4), fmap[1]=c3, fmap[2]=c2, fmap[3]=c1(deepest)
        f_s1, f_s2, f_s3, f_deep = fmap  #

        x = f_deep  # 14×14, c1
        x = self.up3(x, f_s3)  # 28×28
        x = self.up2(x, f_s2)  # 56×56
        x = self.up1(x, f_s1)  # 112×112
        x = self.final_up(x)    # 224×224
        return self.head(x)


class ViT4CNetSkip(nn.Module):
    """ViT encoder + U‑Net decoder """

    def __init__(self, out_channels: int = 4, pretrained: bool = True):
        super().__init__()
        self.encoder = ViTEncoderWithSkips(pretrained=pretrained)
        self.decoder = SkipDecoder(self.encoder.embed_dim, out_channels)
        self.grid_size = self.encoder.grid_size

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips, self.grid_size)


if __name__ == '__main__':
    model = ViT4CNetSkip()
    dummy = torch.randn(2, 3, 224, 224)
    y = model(dummy)
    print('Output shape:', y.shape)  # except [2, 4, 224, 224]
    y.mean().backward()
    print('Grad ok ->', model.encoder.vit.blocks[0].attn.qkv.weight.grad is not None)

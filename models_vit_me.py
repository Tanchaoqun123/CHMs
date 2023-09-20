# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer
from vision_transformer import ConViT


# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
class VisionTransformer(ConViT):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim[-1])

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x_ = self.patch_embed1(x)
        x_ = self.pos_drop(x_)
        for blk in self.blocks1:
            x_ = blk(x_)
        stage1_embed = self.stage1_output_decode(x_).flatten(2).permute(0, 2, 1)
        x_ = self.patch_embed2(x_)
        for blk in self.blocks2:
            x_ = blk(x_)
        stage2_embed = self.stage2_output_decode(x_).flatten(2).permute(0, 2, 1)

        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        stage1_embed = torch.cat((cls_tokens, stage1_embed), dim=1)
        stage2_embed = torch.cat((cls_tokens, stage2_embed), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = x + stage1_embed + stage2_embed

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=[224, 56, 224], patch_size=[4, 2, 16], embed_dim=[256, 384, 768],
        depth=[2, 2, 12], num_heads=12,
        mlp_ratio=[4, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


mae_vit_base_patch16 = vit_base_patch16  # decoder: 512 dim, 8 blocks


if __name__ == '__main__':
    img = torch.randn(16, 3, 224, 224)

    net = vit_base_patch16()
    out = net(img)
    print(out)
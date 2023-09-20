# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from util.pos_embed import get_2d_sincos_pos_embed
# from timm.models.vision_transformer import PatchEmbed, Block, CBlock
from vision_transformer import PatchEmbed, Block, CBlock


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, n_class=14,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.,norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 cls_hidden_mlp=3072, nb_classes=1000, global_pool=True,
                 mlp_depth=2):
        super().__init__()
        # --------------------------------------------------------------------------
        self.image_size = img_size
        # self.patch_size = 4
        self.pos_drop = nn.Dropout()
        self.num_classes = n_class
        self.pool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(decoder_embed_dim, self.num_classes, bias=False)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size=img_size[2], patch_size=patch_size[2], in_chans=in_chans, embed_dim=embed_dim[2])
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        # self.patch_embed3 = PatchEmbed(
        #     img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])

        # self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[2]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim[2]), requires_grad=False)  # fixed sin-cos embedding

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(depth[1])])

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(depth[2])])
        self.norm = norm_layer(embed_dim[2])

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[2], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.pos_drop = nn.Dropout()
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[1])**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Classification branch specifics
        if cls_hidden_mlp == 0:
            self.cls_head = nn.Linear(embed_dim[2], nb_classes)
        else:
            assert mlp_depth in [2], "mlp depth should be 2"
            if mlp_depth == 2:
                self.cls_head = nn.Sequential(
                    nn.Linear(embed_dim[2], cls_hidden_mlp),
                    nn.BatchNorm1d(cls_hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(cls_hidden_mlp, nb_classes),
                )
        self.global_pool = global_pool

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patchify(self, imgs):

        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed.num_patches
        #        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)

        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).\
            permute( 0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).\
            permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)

        x_ = self.patch_embed1(x)
        for blk in self.blocks1:
            x_ = blk(x_, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x_).flatten(2).permute(0, 2, 1)

        x_ = self.patch_embed2(x_)
        for blk in self.blocks2:
            x_ = blk(x_, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x_).flatten(2).permute(0, 2, 1)

        stage1_embed = torch.gather(stage1_embed, dim=1,
                                    index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1,
                                    index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))

        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        # add pos embed w/o cls token
        # x = x + self.pos_embed
        # x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        # append cls token
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) # (16, 50, 512)
        # append mask tokens to sequence
        # 给mask的path附上token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  #(16, 147, 512)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token # (16, 196, 512)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle #(16, 196, 512)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  #(16, 197, 512)

        # add pos embed
        x = x + self.decoder_pos_embed #(16, 197, 512)
        #
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.pos_drop(x)
        # predictor projection
        x = self.decoder_pred(x) #(16, 197, 768)
        # remove cls token
        x = x[:, 1:, :]

        return x  #(16,196,768)

    def forward_classification(self, x):
        if self.global_pool:
            feat = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            feat = x[:, 0, :]  # with cls token
        logits = self.cls_head(feat)
        return logits

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, imgs, mask_ratio=0.75):

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # Classification branch
        logits = self.forward_classification(latent)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask, logits


def mae_vit_me(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=[224, 56, 224], patch_size=[4, 2, 16], embed_dim=[256, 384, 768], depth=[2, 2, 12], num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



# set recommended archs
mae_vit_base_patch16 = mae_vit_me  # decoder: 512 dim, 8 blocks


if __name__ == '__main__':
    img = torch.randn(16, 3, 224, 224)

    net = mae_vit_me()
    out = net(img)
    print(out)
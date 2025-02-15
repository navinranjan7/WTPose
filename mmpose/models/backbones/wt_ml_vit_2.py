# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

import torch.nn.functional as F

from timm.models.layers import DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention

from ..builder import MODELS
from mmcv.cnn import build_norm_layer, build_conv_layer
from ..utils import (MultiheadAttention, SwiGLUFFNFused,
                     resize_pos_embed, to_2tuple)
from .base_backbone import BaseBackbone

### WaterFall Transformer 
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class WTM_A_ML(nn.Module):
    def __init__(self, in_dim, dim, out_dim, kernel_size, num_heads, dilations,  size):
        super(WTM_A_ML, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dilations = dilations
        self.size = size
        
        self.wtm3_0 = NATLayer(dim, num_heads = num_heads[0], kernel_size=kernel_size, dilation = dilations[0])
        self.wtm3_1 = NATLayer(dim, num_heads = num_heads[0], kernel_size=kernel_size, dilation = dilations[1])
        self.wtm2_0 = NATLayer(dim, num_heads = num_heads[1], kernel_size=kernel_size, dilation = dilations[2])
        self.wtm2_1 = NATLayer(dim, num_heads = num_heads[1], kernel_size=kernel_size, dilation = dilations[3])
        self.wtm1_0 = NATLayer(dim, num_heads = num_heads[2], kernel_size=kernel_size, dilation = dilations[4])
        self.wtm1_1 = NATLayer(dim, num_heads = num_heads[2], kernel_size=kernel_size, dilation = dilations[5])
        self.wtm0_0 = NATLayer(dim, num_heads = num_heads[3], kernel_size=kernel_size, dilation = dilations[6])
        self.wtm0_1 = NATLayer(dim, num_heads = num_heads[3], kernel_size=kernel_size, dilation = dilations[7])
        
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_dim),
                                       nn.ReLU())
        # self.conv_3_up = nn.Sequential(nn.ConvTranspose2d(8*dim, 4*dim, kernel_size=2, stride=2),
        #                                nn.BatchNorm2d(4*dim),
        #                                nn.ReLU())
                                    
        self.conv_2_1x1 = nn.Sequential(nn.Conv2d(2*dim, dim, 1),
                                       nn.BatchNorm2d(dim))
        # self.conv_2_up = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        #                                nn.BatchNorm2d(dim),
        #                                nn.ReLU())
        self.conv_1_1x1 = nn.Sequential(nn.Conv2d(2*dim, dim, 1),
                                       nn.BatchNorm2d(dim))
        # self.conv_1_up =  nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        #                                nn.BatchNorm2d(dim),
        #                                nn.ReLU())
        self.conv0_1x1 = nn.Sequential(nn.Conv2d(2*dim, dim, 1),
                                       nn.BatchNorm2d(dim))
        self.init_weights()

    def forward(self, inputs):

        w3 = inputs[3].permute(0,2,3,1)
        w3 = self.wtm3_0(w3)
        w3 = self.wtm3_1(w3)
        w3 = w3.permute(0,3,1,2)
        # w3_up = self.conv_3_up(w3)
        w3_up = F.interpolate(w3, size=(16,12), mode='bilinear', align_corners=True)
        # print(f'w3_up:{w3_up.size()}')
        # print(f'input2:{inputs[2].size()}')

        w2 = torch.cat([inputs[2], w3_up], dim=1)
        w2 = self.conv_2_1x1(w2)
        w2 = w2.permute(0,2,3,1)
        w2 = self.wtm2_0(w2)
        w2 = self.wtm2_1(w2)
        w2 = w2.permute(0,3,1,2)
        # w2_up = self.conv_2_up(w2)
        w2_up = F.interpolate(w2, size=(32,24), mode='bilinear', align_corners=True)
        # print(f'w2_up:{w2_up.size()}')
        # print(f'input1:{inputs[1].size()}')

        w1 = torch.cat([inputs[1], w2_up], dim=1)
        w1 = self.conv_1_1x1(w1)
        w1 = w1.permute(0,2,3,1)
        w1 = self.wtm1_0(w1)
        w1 = self.wtm1_1(w1)
        w1 = w1.permute(0,3,1,2)
        # w1_up = self.conv_1_up(w1)
        w1_up = F.interpolate(w1, size=(64,48), mode='bilinear', align_corners=True)
        # print(f'w1_up:{w1_up.size()}')
        # print(f'input0:{inputs[0].size()}')

        w0 = torch.cat([inputs[0], w1_up], dim=1)
        w0 = self.conv0_1x1(w0)
        w0 = w0.permute(0,2,3,1)
        w0 = self.wtm0_0(w0)
        w0 = self.wtm0_1(w0)
        w0 = w0.permute(0,3,1,2)

        x = self.last_conv(w0)
        # print(f'wtm_out:{x.size()}')

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)



class Connect(nn.Module):
    def __init__(self, in_channels=768, out_channels=128, ratio=1):
        super(Connect, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio

        hidden_features = self.in_channels // self.ratio

        # Define layers for upsampling/downsampling based on the up_step
        self.conn_1 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, hidden_features, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_features, self.out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(hidden_features),
        #     nn.ReLU()
        # )
        # self.conv_transpose_2 = nn.Sequential(
        #     nn.ConvTranspose2d(hidden_features, self.out_channels, kernel_size=2, stride=2, padding=0),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU()
        # )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU()
        # )
        self.conn_2 = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU()
        # )
        self.conn_3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.conn_4 = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_features, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.Conv2d(hidden_features, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.init_weights()

    def forward(self, inputs):
        c1 = self.conn_1(inputs[0])
        c2 = self.conn_2(inputs[1])
        c3 = self.conn_3(inputs[2])
        c4 = self.conn_4(inputs[3])

        # print(f'c1_: {c1.size()}')
        # c2 = self.forward_step(inputs[1], up_step=1)
        # print(f'c2_: {c2.size()}')
        # c3 = self.forward_step(inputs[2], up_step=0)
        # print(f'c3_: {c3.size()}')
        # c4 = self.forward_step(inputs[3], up_step="downsample")
        # print(f'c4_: {c4.size()}')
        return [c1, c2, c3, c4]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='origin',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)[1]

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value)

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)[1]

        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value)
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x


@MODELS.register_module()
class WT_ML_ViTV2(BaseBackbone):
    """Waterfall Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16. Pose: 256x192 = 16x12.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"cls_token"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small', 'dinov2-s', 'dinov2-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
        **dict.fromkeys(
            ['dinov2-g', 'dinov2-giant'], {
                'embed_dims': 1536,
                'num_layers': 40,
                'num_heads': 24,
                'feedforward_channels': 6144
            }),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 wtm_in_dim,
                 wtm_dim,
                 wtm_out_dim,
                 wtm_kernel_size,
                 wtm_num_heads,
                 wtm_dilations,
                 wtm_size,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 con_in_channels=768,
                 con_out_channels=128,
                 con_ratio=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='cls_token',
                 with_cls_token=True,
                 frozen_stages=0,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None):
        super(WT_ML_ViTV2, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != 'cls_token':
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.frozen_stages = frozen_stages
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)[1]
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)[1]

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()
        _connect_cfg = {
            'in_channels':con_in_channels,
            'out_channels':con_out_channels,
            'ratio':con_ratio
        }
        self.connect = Connect(**_connect_cfg)
        _wtm_cfg = {
            'in_dim': wtm_in_dim,
            'dim': wtm_dim,
            'out_dim': wtm_out_dim,
            'kernel_size': wtm_kernel_size,
            'num_heads': wtm_num_heads,
            'dilations': wtm_dilations,
            'size': wtm_size
            }
        self.wtm_ml = WTM_A_ML(**_wtm_cfg)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(WT_ML_ViTV2, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if (not self.with_cls_token
                and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
            # Remove cls token from state dict if it's not used.
            state_dict[name] = state_dict[name][:, 1:]
            ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        # print(f'input_x: {x.size()}')
        x, patch_resolution = self.patch_embed(x)
        patch_resolution = patch_resolution[::-1]
        # print(patch_resolution)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

            # print(f'con1:{outs[0].size()}')
        connect = self.connect(tuple(outs))
        wtm = self.wtm_ml(connect)
        return wtm


    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from copy import deepcopy
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmengine.runner import load_state_dict

import torch.nn.functional as F

from timm.models.layers import DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention

from ..builder import MODELS
# from mmpose.utils import get_root_logger
# from mmcv.cnn import build_norm_layer
from ..utils import (MultiheadAttention, SwiGLUFFNFused,
                     resize_pos_embed, to_2tuple)
from .base_backbone import BaseBackbone
from ..utils import BasicBlock, Bottleneck, ResLayer
# from ..utils import WTM_A_1

#ResNet Bottleneck

def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion

class ResNet(BaseBackbone):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default:
            ``[
                dict(type='Kaiming', layer=['Conv2d']),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]``

    Example:
        >>> from mmpose.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18, out_indices=(0, 1, 2, 3))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23,)),
        # 101: (Bottleneck, (3, 4, 23,3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=101,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=3,
                 strides=(1,2,2,),
                 dilations=(1,1,1,),
                 out_indices=(0, 2, ),
                 style='pytorch',
                 deep_stem=True,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super(ResNet, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)


        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels
        
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

    def make_res_layer(self, **kwargs):
        """Make a ResLayer."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer."""
        if self.deep_stem:
            print('deep_stem')
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                # ConvModule(
                #     stem_channels // 2,
                #     stem_channels // 2,
                #     kernel_size=3,
                #     stride=1,
                #     padding=1,
                #     conv_cfg=self.conv_cfg,
                #     norm_cfg=self.norm_cfg,
                #     inplace=True),
                ConvModule(
                    stem_channels,
                    stem_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            print('not deep_stem')
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone."""
        super(ResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        # x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # print(f'x:{x.size()}')
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

## Transformer connecter
class UpSample(nn.Module):
    def __init__(self, in_channels=768, out_channels=128, ratio=1, norm_cfg=None):
        super(UpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        hidden_features = self.in_channels // self.ratio
        self.conv_transpose_1 = nn.Sequential(
                                nn.ConvTranspose2d(in_channels, hidden_features, kernel_size=2, stride=2, padding=0),
                                # build_norm_layer(norm_cfg, hidden_features)[1],
                                nn.BatchNorm2d(hidden_features),
                                nn.ReLU())
        self.conv_1 = nn.Sequential(
                        nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1),
                        # build_norm_layer(norm_cfg, hidden_features)[1],
                        nn.BatchNorm2d(hidden_features),
                        nn.ReLU())
        self.conv_transpose_2 = nn.Sequential(
                                nn.ConvTranspose2d(hidden_features, out_channels, kernel_size=2, stride=2, padding=0),
                                # build_norm_layer(norm_cfg, out_channels)[1],
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU())
        self.conv_2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        # build_norm_layer(norm_cfg, out_channels)[1],
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.con_init_weights()
    
    def forward(self, x):
        x = self.conv_transpose_1(x)
        x = self.conv_1(x)
        x = self.conv_transpose_2(x)
        x = self.conv_2(x)
        return x
    
    def con_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
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
    def __init__(self, in_channels=768, out_channels=128, patch_size=16, ratio=1, norm_cfg=None):
        super(Connect, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.ratio = ratio

        self.c1 = UpSample(in_channels, out_channels, ratio, norm_cfg)
        self.c2 = UpSample(in_channels, out_channels, ratio, norm_cfg)
        self.c3 = UpSample(in_channels, out_channels, ratio, norm_cfg)
        self.c4 = UpSample(in_channels, out_channels, ratio, norm_cfg)

    # def to_2D(self, x):
    #     # print(f'c1:{x.size()}')
    #     n, hw, c = x.shape
    #     if len(self.patch_size) == 2:
    #         h = self.patch_size[0]
    #         w = self.patch_size[1]
    #     else:
    #         h = w = int(math.sqrt(hw))
    #     x = x.transpose(1,2).reshape(n, c, h, w)
    #     # print(f'to_2D: {x.size()}')
    #     return x

    def forward(self, inputs):
        c1 = self.c1(inputs[0])
        c2 = self.c2(inputs[1])
        c3 = self.c3(inputs[2])
        c4 = self.c4(inputs[3])
        return [c1, c2, c3, c4]

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

class WTM_A_1(nn.Module):
    def __init__(self, in_dim, dim, out_dim, kernel_size, num_heads, dilations,  size, low_level_dim, reduction):
        super(WTM_A_1, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dilations = dilations
        self.size = size
        self.low_level_dim = low_level_dim
        self.reduction = reduction
        
        # convs = conv_dict[conv_type]
        

        # self.conv_expand = nn.Conv2d(out_dim, dim, 1, bias=False)
        # self.bn_expand = nn.BatchNorm2d(dim)
        self.wtm1_0 = NATLayer(dim, num_heads = num_heads[0], kernel_size=kernel_size, dilation = dilations[0])
        self.wtm1_1 = NATLayer(dim, num_heads = num_heads[0], kernel_size=kernel_size, dilation = dilations[1])
        self.wtm2_0 = NATLayer(dim, num_heads = num_heads[1], kernel_size=kernel_size, dilation = dilations[2])
        self.wtm2_1 = NATLayer(dim, num_heads = num_heads[1], kernel_size=kernel_size, dilation = dilations[3])
        self.wtm3_0 = NATLayer(dim, num_heads = num_heads[2], kernel_size=kernel_size, dilation = dilations[4])
        self.wtm3_1 = NATLayer(dim, num_heads = num_heads[2], kernel_size=kernel_size, dilation = dilations[5])
        self.wtm4_0 = NATLayer(dim, num_heads = num_heads[3], kernel_size=kernel_size, dilation = dilations[6])
        self.wtm4_1 = NATLayer(dim, num_heads = num_heads[3], kernel_size=kernel_size, dilation = dilations[7])
        
        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(dim, dim, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(dim),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(in_dim, dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        
        self.conv2 = nn.Conv2d(5*dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        
        self.low = nn.Conv2d(self.low_level_dim, self.reduction, 1, bias=False)
        self.bn_low = nn.BatchNorm2d(self.reduction)

        self.last_conv = nn.Sequential(nn.Conv2d(dim+ self.reduction, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_dim),
                                       nn.ReLU())
        self.init_weights()

    def forward(self, inputs, low_level_feat):
        
        # inputs1 = F.interpolate(inputs[1], size=self.size, mode='bilinear', align_corners=True)
        # inputs2 = F.interpolate(inputs[2], size=self.size, mode='bilinear', align_corners=True)
        # inputs3 = F.interpolate(inputs[3], size=self.size, mode='bilinear', align_corners=True)			
        x = torch.cat([inputs[0], inputs[1], inputs[2], inputs[3]], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        
        x_per = x.permute(0,2,3,1)
        # print(f'x: {x.size()}')
        
        x1 = self.wtm1_0(x_per)
        x1 = self.wtm1_1(x1)
        # print(f'1:{x1.size()}')
        
        x2 = self.wtm2_0(x1)
        x2 = self.wtm2_1(x2) 
        # print(f'2:{x2.size()}')
        
        x3 = self.wtm3_0(x2)
        x3 = self.wtm3_1(x3)
        # print(f'3:{x3.size()}')
        
        x4 = self.wtm4_0(x3)
        x4 = self.wtm4_1(x4)
        # print(f'4:{x4.size()}')
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[1:-1], mode='bilinear', align_corners=True)
        x5 = x5.permute(0,2,3,1)
        # print(f'5:{x5.size()}')
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=3)
        x = x.permute(0,3,1,2)
        
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        low_level_feat = self.low(low_level_feat)
        low_level_feat = self.bn_low(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = torch.cat([x, low_level_feat], dim=1)

        x = self.last_conv(x)
        # print(f'wtm_out:{x.size()}')

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
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

####

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
class ResnetWViT(BaseBackbone):
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
                 wtm_low_level_dim,
                 wtm_reduction,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 con_patch_size=(16,12),
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
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None):
        super(ResnetWViT, self).__init__(init_cfg)

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
        # self.patch_embed = PatchEmbed(**_patch_cfg)
        # self.patch_resolution = self.patch_embed.init_out_size
        # num_patches = self.patch_resolution[0] * self.patch_resolution[1]

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
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches + self.num_extra_tokens,
        #                 self.embed_dims))
        # self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

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
            'patch_size': con_patch_size,
            'ratio':con_ratio,
            'norm_cfg':norm_cfg
        }
        self.connect = Connect(**_connect_cfg)
        _wtm_cfg = {
            'in_dim': wtm_in_dim,
            'dim': wtm_dim,
            'out_dim': wtm_out_dim,
            'kernel_size': wtm_kernel_size,
            'num_heads': wtm_num_heads,
            'dilations': wtm_dilations,
            'size': wtm_size,
            'low_level_dim': wtm_low_level_dim,
            'reduction':wtm_reduction
            }
        self.wtm_a = WTM_A_1(**_wtm_cfg)
        self.resnet = ResNet()
        self.resnet_1x1 = ConvModule(
                    1024,
                    768,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    inplace=True)
        # self.resnet_patch = ResNet_patch()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(ResnetWViT, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    # def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
    #     name = prefix + 'pos_embed'
    #     if name not in state_dict.keys():
    #         return

    #     ckpt_pos_embed_shape = state_dict[name].shape
    #     if (not self.with_cls_token
    #             and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
    #         # Remove cls token from state dict if it's not used.
    #         state_dict[name] = state_dict[name][:, 1:]
    #         ckpt_pos_embed_shape = state_dict[name].shape

    #     if self.pos_embed.shape != ckpt_pos_embed_shape:
    #         from mmengine.logging import MMLogger
    #         logger = MMLogger.get_current_instance()
    #         logger.info(
    #             f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
    #             f'to {self.pos_embed.shape}.')

    #         ckpt_pos_embed_shape = to_2tuple(
    #             int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
    #         pos_embed_shape = self.patch_embed.init_out_size

            # state_dict[name] = resize_pos_embed(state_dict[name],
            #                                     ckpt_pos_embed_shape,
            #                                     pos_embed_shape,
            #                                     self.interpolate_mode,
            #                                     self.num_extra_tokens)

    @staticmethod
    # def resize_pos_embed(*args, **kwargs):
    #     """Interface for backward-compatibility."""
    #     return resize_pos_embed(*args, **kwargs)

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
        resnet_out_ = self.resnet(x)
        resnet_out = resnet_out_[0]
        # print(f'Resnet" {resnet_out.shape}')
        x = self.resnet_1x1(resnet_out_[1]) 
        # print(f'resnet_out2: {resnet_out.size()}')
        x= x.permute(0, 2, 3, 1)
        patch_resolution = x.shape[1], x.shape[2]
        x = x.reshape(-1, x.shape[1]*x.shape[2],x.shape[3])
        # print(f'resnet_out3: {resnet_out.size()}')
        
        # x, patch_resolution = self.patch_embed(x)2
        # print(x.size())s
        # print(patch_resolution.size())

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # x = x + resize_pos_embed(
        #     self.pos_embed,
        #     self.patch_resolution,
        #     patch_resolution,
        #     mode=self.interpolate_mode,
        #     num_extra_tokens=self.num_extra_tokens)
        # x = self.drop_after_pos(x)

        x = self.pre_norm(x)
        # print(x.size())

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

            # print(f'con1:{outs[0].size()}')
        connect = self.connect(tuple(outs))
        wtm = self.wtm_a(connect, resnet_out)
        return wtm

        # return tuple(outs)

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
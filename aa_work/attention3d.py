#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:59:12 2023

@author: nr4325
"""

import torch
from torch import nn

from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
from patch3d import window_3d_partition, window_3d_partition_join

#%%
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

class WindowAttention3D(nn.Module):
    """ Group based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted group.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Gd, Gh, Gw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nG = mask.shape[0]
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
#%%
class WaterfallTransformerLayer3D(nn.Module):
    """ Multi-head Waterfall Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        dilation (tuple[int]): Dilation rate for D-WTB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(3,3,3), dilation=(1,1,1), stride=(1,1,1),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.stride = stride
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        x_shape = x.shape

        x = self.norm1(x)
        
        # partition windows
        x, dhw, pad = window_3d_partition(x, self.window_size, self.dilation, self.stride)
        x = self.attn(x)
        x = window_3d_partition_join(x_shape, x, dhw, self.window_size, self.dilation, self.stride, padding=pad)
        print(x)
        return x
    
    def forward_part2(self, x):
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x
    
    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
            
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x
#%%
a = torch.arange(1, 128*96*72*15+1, dtype=torch.float).view(1,15,96,72,128)
# a = torch.rand(1, 10, 24, 24, 32)
print(a.shape)
xx1 =WaterfallTransformerLayer3D(dim=128, num_heads=4, window_size=(3,7,7), dilation=(1,1,1), stride=(3,7,7))
xx2 =WaterfallTransformerLayer3D(dim=128, num_heads=4, window_size=(3,7,7), dilation=(8,8,8), stride=(3,7,7))
b = xx1(a)
print(b.shape)
c = xx2(b)
print(c.shape)
#print(b)


# b = window_3d_partition(a, kernel_size=(3,3,3), stride=1, dilation=(1,1,1))

# # # print(b[0])
#print(b)
#print(b[-1])
# print(b.shape)
# c = window_3d_merge(b, (3,3,3), (1,1,1), dhw_dims, channels)
#%%

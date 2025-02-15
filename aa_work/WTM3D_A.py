#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:56:49 2023

@author: nr4325
"""
import torch
from torch import nn

from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
from patch3d import window_3d_partition, window_3d_partition_join
import torch.nn.functional as F

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
class WTM3D(nn.Module):
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
    def __init__(self, in_dim, dim, out_dim, window_size, num_heads, dilations, stride, size, low_level_dim, reduction):
        super(WTM3D, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dilations = dilations
        self.stride = stride
        self.size = size
        self.low_level_dim = low_level_dim
        self.reduction = reduction
        
        # convs = conv_dict[conv_type]
        

        # self.conv_expand = nn.Conv2d(out_dim, dim, 1, bias=False)
        # self.bn_expand = nn.BatchNorm2d(dim)
        self.wtb1_0 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[0], window_size=window_size, dilation = dilations[0], stride=stride)
        self.wtb1_1 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[0], window_size=window_size, dilation = dilations[1], stride=stride)

        self.wtb2_0 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[1], window_size=window_size, dilation = dilations[2], stride=stride)
        self.wtb2_1 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[1], window_size=window_size, dilation = dilations[3], stride=stride)

        self.wtb3_0 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[2], window_size=window_size, dilation = dilations[4], stride=stride)
        self.wtb3_1 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[2], window_size=window_size, dilation = dilations[5], stride=stride)

        self.wtb4_0 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[3], window_size=window_size, dilation = dilations[6], stride=stride)
        self.wtb4_1 = WaterfallTransformerLayer3D(dim, num_heads = num_heads[3], window_size=window_size, dilation = dilations[7], stride=stride)
        
        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_dim, dim, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(dim),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(in_dim, dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        
        self.conv2 = nn.Conv2d(61*dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        
        self.low = nn.Conv2d(low_level_dim, reduction, 1, bias=False)
        self.bn_low = nn.BatchNorm2d(reduction)

        self.last_conv = nn.Sequential(nn.Conv2d(dim+reduction, dim, kernel_size=3, stride=1, padding=1, bias=False),
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
        # x_ = torch.cat([inputs[0], inputs1, inputs2, inputs3], dim=1)
        
        x = inputs.view(-1, 15, self.size[0], self.size[1], self.dim)
        
        #// Need to adjust here for 3D
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        
        
        
        # x_per = x.permute(0,2,3,1) 
        # print(f'x: {x.size()}')
        
        x1 = self.wtb1_0(x)
        x1 = self.wtb1_1(x1)
        print(f'1:{x1.size()}')
        
        x2 = self.wtb2_0(x1)
        x2 = self.wtb2_1(x2) 
        print(f'2:{x2.size()}')
        
        x3 = self.wtb3_0(x2)
        x3 = self.wtb3_1(x3)
        print(f'3:{x3.size()}')
        
        x4 = self.wtb4_0(x3)
        x4 = self.wtb4_1(x4)
        print(f'4:{x4.size()}')
        
        x_ = inputs.permute(0, 3, 1, 2)
        x5 = self.global_avg_pool(x_)
        x5 = F.interpolate(x5, size=x4.size()[2:-1], mode='bilinear', align_corners=True)
        x5 = torch.unsqueeze(x5, 1)
        x5 = x5.permute(0, 1, 3, 4, 2)
        # x5 = x5.permute(0,2,3,1)
        print(f'5:{x5.size()}')
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = x.permute(0,2,3,4,1)

        x = x.reshape(-1, x.shape[1], x.shape[2], x.shape[3]*x.shape[4])
        print(f'cat2:{x.size()}')
        x = x.permute(0,3,1,2)
        
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        print(f'con2:{x.size()}')
        
        low_level_feat = self.low(low_level_feat)
        low_level_feat = self.bn_low(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = torch.cat([x, low_level_feat], dim=1)

        x = self.last_conv(x)
        print(f'last:{x.size()}')

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
#%%
def main():
    a = torch.arange(1, 2*128*9*7*15+1, dtype=torch.float).view(2,9,7,128*15)
    b = torch.arange(1, 2*256*9*7+1, dtype=torch.float).view(2,256,9,7)
    # a = torch.rand(1, 10, 24, 24, 32)
    # print(a.shape)
    xx = WTM3D(in_dim=1920,   # sum of all channels
            window_size = (7, 7, 7),
            dim=128, 
            out_dim=128,
            num_heads = [8, 8, 8, 8],
            dilations = [(2,2,2), (1,1,1),(4,4,4),(1,1,1),(6,6,6),(1,1,1),(8,8,8),(1,1,1)],
            stride = (7,7,7),
            size=(9, 7),
            low_level_dim=256,
            reduction=32)
    aa = xx(a, b)
    # xx1 =WaterfallTransformerLayer3D(dim=128, num_heads=4, window_size=(3,7,7), dilation=(1,1,1), stride=(3,7,7))
    # xx2 =WaterfallTransformerLayer3D(dim=128, num_heads=4, window_size=(3,7,7), dilation=(8,8,8), stride=(3,7,7))
    # b = xx1(a)
    # print(b.shape)
    # c = xx2(b)
    # print(c.shape)
#%%
if __name__== '__main__':
    main()
#%%
# a1 = torch.unsqueeze(a, 1)
# print(a1.shape)
#%%
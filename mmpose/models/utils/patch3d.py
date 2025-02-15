#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:16:07 2023

@author: nr4325
"""
import torch
from torch import nn

def window_3d(x, kernel_size, dilation, stride=1):

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out
    def get_padding(kernel_size, dilation):
        pad = []
        for i in range(len(list(kernel_size))):
            pad_i = dilation[i]*(kernel_size[i]-1)//2
            pad.append(pad_i)
        return pad

    channels = x.shape[1]
    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    padding = get_padding(kernel_size, dilation)
    d_dim_out = get_dim_blocks(d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)
    
    # (B, C, D, H, W)
    x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)                                                     
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, 
                                    kernel_size=(kernel_size[0], 1), 
                                    padding=(padding[0], 0), 
                                    stride=(stride[0], 1), 
                                    dilation=(dilation[0], 1))                   
    # # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)                                   
    # # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, 
                                    kernel_size=(kernel_size[1], kernel_size[2]), 
                                    padding=(padding[1], padding[2]), 
                                    stride=(stride[1], stride[2]), 
                                    dilation=(dilation[1], dilation[2]))        
    # # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)
    

    x = x.view(-1, channels, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    # # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    
    x = x.contiguous().view(-1, kernel_size[0]*kernel_size[1]*kernel_size[2], channels)
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])


    return x

def window_3d_partition(x, kernel_size, dilation, stride=1):

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    def get_padding(kernel_size, dilation):
        pad = []
        for i in range(len(list(kernel_size))):
            if dilation[i] == 1:
                pad_i = 0
            else:
                pad_i = dilation[i]*(kernel_size[i]-1)//2
            pad.append(pad_i)
        return pad

    B, D, H, W, C = x.shape
    
    padding = get_padding(kernel_size, dilation)
    # print(padding)
    
    d_dim_out = get_dim_blocks(D, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_out = get_dim_blocks(H, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_out = get_dim_blocks(W, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_out, h_dim_out, w_dim_out)

    x = x.view(-1, C, D, H * W)                                                     
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))                   
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, C * kernel_size[0] * d_dim_out, H, W)                                   
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))        
    # # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)
    

    x = x.view(-1, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    # # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    
    x = x.permute(0, 3, 6, 7, 1, 2, 4, 5)
    # (B, d_dim_out, h_dim_out, w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    
    x = x.contiguous().view(-1, C, kernel_size[0], kernel_size[1], kernel_size[2])

    
    x = x.contiguous().view(-1, kernel_size[0]*kernel_size[1]*kernel_size[2],C)
    # (B * d_dim_out * h_dim_out * w_dim_out, kernel_size[0]*kernel_size[1]*kernel_size[2],C)

    return [x, [d_dim_out, h_dim_out, w_dim_out], padding] 
    # return x

def window_3d_partition_join(original_dim, x, dhw_size, kernel_size, dilation, stride, padding):
    
    k0, k1, k2 = kernel_size
    d,h,w = dhw_size
    p0, p1, p2 = padding
    d0, d1, d2 = dilation
    s0, s1, s2 = stride
    B,D,H,W,C = original_dim
    
    
    x1 = x.view(-1,k0,k1,k2,C)
    # print(x1.shape)
    x2 = x1.view(-1, d,h,w,k0,k1,k2,C)
    # print(x2.shape)
    x3 = x2.permute(0,7,4,1,5,6,2,3)
    # print(x3.shape)


    x4 = x3.contiguous().view(-1,C*k0*d*k1*k2,h*w)
    # print(x4.shape)
    fold = nn.Fold(output_size=(H,W), kernel_size=(k1,k2), stride =(s1, s2), padding=(p1, p2), dilation=(d1, d2))
    x5 = fold(x4)
    # print(x5.shape)


    x6 = x5.view(-1, C*k0, d*H*W)
    # print(x6.shape)
    fold1 = nn.Fold(output_size=(D, H*W), kernel_size=(k0,1), stride =(s0, 1), padding=(p0, 0), dilation=(d0, 1))

    x7 = fold1(x6)
    # print(x7.shape)
    x8 = x7.view(-1, C, D, H, W)
    # print(x8.shape)
    x9 = x8.view(-1, D, H, W, C)
    # print(x9.shape)
    return x9
#%%
# a = torch.arange(1, 128*96*72*15+1, dtype=torch.float).view(1,15,96,72,128)
# #B,C,D,H,W
# kernel_size=(3,7,7)
# stride=(3,3,3)
# dilation=(2,2,2)

# print(a.shape)
# b, dhw, pad  = window_3d_partition(a, kernel_size=kernel_size, stride=stride, dilation=dilation)

# # print(b[0])
# #print(b)

# print(b.shape)
# #%%
# b_j = window_3d_partition_join(a.shape, b, dhw, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=pad)

#%%

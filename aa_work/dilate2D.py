#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:49:35 2023

@author: nr4325
"""

import torch 
from natten import NeighborhoodAttention3D
# x = torch.randint(1, 100, (1,1,6,4))
# B, C, H, W = x.shape
# window_size = 3
# window_size_d = 4
# print(x)
#%%
##3D patch selection
# x = torch.randint(1, 10, (1,1,4,4,4))
# B, C, D, H, W = x.shape
# window_size = 2
# window_size_d = 4
# x = x.view(B, D//window_size_d, window_size_d, H // window_size, window_size, W // window_size, window_size, C)
# x = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, window_size_d, window_size,  window_size, C)
# # x=x.view(-1, window_size, window_size,C)
# print(x[0])
#%%
## 2D Dilated patch selection
# x = torch.randint(1, 100, (1,1,4,4))
# B, C, H, W = x.shape
# window_size = 2
# print(x)

# x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
# x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B*window_size*window_size, window_size,window_size, C)

# print(x.shape)
# print(x[0])
# print(x[0])
# print(x[3])
#%%
## 2D Dilated patch selection 3x3 window (self dilation of 3)???
# x = torch.randint(1, 100, (1,1,9,9))
# B, C, H, W = x.shape
# window_size = 3
# print(x)
# x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
# # x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B*window_size*window_size, window_size, window_size, C)

# print(x.shape)
# print(x[0])
# print(x[0])
# print(x[3])
#%%
## 2D Dilated patch selection 3x3 window
# x = torch.randint(1, 100, (1,1,12,12))
# B, C, H, W = x.shape
# window_size = 3
# dilation = 2

# # interval_h = H//3
# # interval_w = W//3

# # Hd = H // interval_h
# # Wd = W // interval_w
# print(x)
# x = x.view(B, H//2 , 2, W//2, 2, C)
# # x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
# # x = x.reshape(B*interval_h*interval_w, Hd*Wd, C)

# # x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B*window_size*window_size, window_size, window_size, C)

# print(x.shape)
# print(x[0])
# # print(x[0])
# print(x[3])
#%%
## 2D Dilated patch selection 3x3 window
x = torch.randint(1, 100, (1,1, 12, 12))
print(x)
B, C, H, W= x.shape
# Fix window 
x = x.view(1,H//3, 3, W//3, 3, C)
x = x.permute(0, 1, 3,2,4, 5)
# x = x.reshape(1, 4,3)
print(x.shape)
print(x[0])
#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:53:52 2023

@author: nr4325
"""

import torch
from torch import nn

a = torch.arange(1, 15*96*72*128+1, dtype=torch.float).view(1,15,96,72,128)
print(a.shape)
b = torch.arange(1, 3840*343*128+1, dtype=torch.float).view(3840, 343, 128)
print(b.shape)
k0, k1, k2 = 7,7,7
p0, p1, p2 = 6, 6, 6
d0, d1, d2 = 2,2,2
d,h,w = 5,32,24
B,D,H,W,C = a.shape
s0, s1, s2 = 3,3,3

b1 = b.view(-1,k0,k1,k2,C)
print(b1.shape)
b2 = b1.view(-1, d,h,w,k0,k1,k2,C)
print(b2.shape)
b3 = b2.permute(0,7,4,1,5,6,2,3)
print(b3.shape)


b4 = b3.contiguous().view(-1,C*k0*d*k1*k2,h*w)
print(b4.shape)
fold = nn.Fold(output_size=(H,W), kernel_size=(k1,k2), stride =(s1, s2), padding=(p1, p2), dilation=(d1, d2))
b5 = fold(b4)
print(b5.shape)


b6 = b5.view(-1, C*k0, d*H*W)
print(b6.shape)
fold1 = nn.Fold(output_size=(D, H*W), kernel_size=(k0,1), stride =(s0, 1), padding=(p0, 0), dilation=(d0, 1))

b7 = fold1(b6)
print(b7.shape)
b8 = b7.view(-1, C, D, H,W)
print(b8.shape)
b9 = b8.view(-1, D, H,W,C)

print(b9.shape)

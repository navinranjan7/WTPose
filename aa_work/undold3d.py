#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:44:52 2023

@author: nr4325
"""

import torch
input = torch.randn(2, 3, 4, 4,5)

output = torch.nn.functional.unfold(input, kernel_size=(3, 3,3))

print(output.shape)
#%%
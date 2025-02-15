import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS

from timm.models.layers import DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention

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

@NECKS.register_module()
class WTM_A(nn.Module):
    def __init__(self, in_dim, dim, out_dim, kernel_size, num_heads, dilations, size, low_level_dim, reduction_ratio):
        super(WTM_A, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dilations = dilations
        self.size = size
        self.low_level_dim = low_level_dim
        self.reduction_ratio = reduction_ratio
        
        # convs = conv_dict[conv_type]
        

        # self.conv_expand = nn.Conv2d(out_dim, dim, 1, bias=False)
        # self.bn_expand = nn.BatchNorm2d(dim)
        self.twasp1_0 = NATLayer(dim, num_heads = num_heads[0], kernel_size=kernel_size, dilation = dilations[0])
        self.twasp1_1 = NATLayer(dim, num_heads = num_heads[0], kernel_size=kernel_size, dilation = dilations[1])

        self.twasp2_0 = NATLayer(dim, num_heads = num_heads[1], kernel_size=kernel_size, dilation = dilations[2])
        self.twasp2_1 = NATLayer(dim, num_heads = num_heads[1], kernel_size=kernel_size, dilation = dilations[3])

        self.twasp3_0 = NATLayer(dim, num_heads = num_heads[2], kernel_size=kernel_size, dilation = dilations[4])
        self.twasp3_1 = NATLayer(dim, num_heads = num_heads[2], kernel_size=kernel_size, dilation = dilations[5])

        self.twasp4_0 = NATLayer(dim, num_heads = num_heads[3], kernel_size=kernel_size, dilation = dilations[6])
        self.twasp4_1 = NATLayer(dim, num_heads = num_heads[3], kernel_size=kernel_size, dilation = dilations[7])
        
        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(dim, dim, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(dim),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(in_dim, dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        
        self.conv2 = nn.Conv2d(5*dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        
        reduction = low_level_dim // reduction_ratio
        
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
        
        inputs1 = F.interpolate(inputs[1], size=self.size, mode='bilinear', align_corners=True)
        inputs2 = F.interpolate(inputs[2], size=self.size, mode='bilinear', align_corners=True)
        inputs3 = F.interpolate(inputs[3], size=self.size, mode='bilinear', align_corners=True)			
        x = torch.cat([inputs[0], inputs1, inputs2, inputs3], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        
        x_per = x.permute(0,2,3,1)
        # print(f'x: {x.size()}')
        
        x1 = self.twasp1_0(x_per)
        x1 = self.twasp1_1(x1)
        # print(f'1:{x1.size()}')
        
        x2 = self.twasp2_0(x1)
        x2 = self.twasp2_1(x2) 
        # print(f'2:{x2.size()}')
        
        x3 = self.twasp3_0(x2)
        x3 = self.twasp3_1(x3)
        # print(f'3:{x3.size()}')
        
        x4 = self.twasp4_0(x3)
        x4 = self.twasp4_1(x4)
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

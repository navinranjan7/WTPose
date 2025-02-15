# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .cpm import CPM
from .hourglass import HourglassNet
from .hourglass_ae import HourglassAENet
from .hrformer import HRFormer
from .hrnet import HRNet
from .litehrnet import LiteHRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mspn import MSPN
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .rsn import RSN
from .scnet import SCNet
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin import SwinTransformer
from .swinv2 import SwinV2
from .tcn import TCN
from .v2v_net import V2VNet
from .vgg import VGG
from .vipnas_mbv3 import ViPNAS_MobileNetV3
from .vipnas_resnet import ViPNAS_ResNet
from .WTPose_SwinV2 import WTPoseSwinV2
from .wt_swinv2  import wtSwinV2
from .wt_swin import wtSwin
from .wt_swinv2_wtm3d import wtSwinV2_wtm3d
from .wtpose_swinv1_wtm3a import wtposeSwinV1WTM3
from .wtpose_swinv1_wtm3L import wtposeSwinV1WTM3L
from .vit import ViT
from .vision_transformer import VisionTransformer
from .wvit import WViT
from .resnet_wvit import ResnetWViT
from .resnet_wvit_BC import ResnetWViTBC
from .wt_ml_vit import WT_ML_ViT
from .wt_ml_vit_2 import WT_ML_ViTV2
from .wt_ml_all_vit import WT_ML_ALL_ViTV2
from .wt_ml_all_f_vit import WT_ML_ALL_F_ViTV2
from .wt_vit import WTM_ViT
from .wt_all_vit import WTM_ALL_ViT
from .wtpose import WTPose
from .wtpose_ml import WTPose_MLv1
from .wtpose_mlv2 import WTPose_MLv2
from .wtpose_mlv2_192 import WTPose_MLv2_192
from .wtpose_mlv3 import WTPose_MLv3

__all__ = [
    'AlexNet', 'HourglassNet', 'HourglassAENet', 'HRNet', 'MobileNetV2',
    'MobileNetV3', 'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SCNet',
    'SEResNet', 'SEResNeXt', 'ShuffleNetV1', 'ShuffleNetV2', 'CPM', 'RSN',
    'MSPN', 'ResNeSt', 'VGG', 'TCN', 'ViPNAS_ResNet', 'ViPNAS_MobileNetV3',
    'LiteHRNet', 'V2VNet', 'HRFormer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'SwinTransformer', 'SwinV2', 'WTPoseSwinV2',
    'wtSwinV2', 'wtSwin', 'wtSwinV2_wtm3d', 'wtposeSwinV1WTM3', 'wtposeSwinV1WTM3L' ,'ViT', 'VisionTransformer', 'WViT', 'ResnetWViT', 'ResnetWViTBC', 'WT_ML_ViT',
    'WT_ML_ViTV2', 'WT_ML_ALL_ViTV2', 'WT_ML_ALL_F_ViTV2', 'WTM_ViT', 'WTM_ALL_ViT', 'WTPose', 'WTPose_MLv1', 'WTPose_MLv2', 'WTPose_MLv2_192', 'WTPose_MLv3'
]

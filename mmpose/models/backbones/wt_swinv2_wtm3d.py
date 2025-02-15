from copy import deepcopy
from typing import Sequence

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList, constant_init
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from ..builder import MODELS
from ..utils import PatchMerging, ShiftWindowMSA, WindowMSAV2, resize_pos_embed, to_2tuple
# from ..utils import (resize_pos_embed, to_2tuple)
from .base_backbone import BaseBackbone
from ..utils import BasicBlock, Bottleneck, ResLayer
# from ..utils import WTM3D_A

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
        101: (Bottleneck, (3,)),
        152: (Bottleneck, (3,))
    }

    def __init__(self,
                 depth=152,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=1,
                 strides=(1,),
                 dilations=(1,),
                 out_indices=(0, ),
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
    
class SwinBlockV2(BaseModule):
    """Swin Transformer V2 block. Use post normalization.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        shift (bool): Shift the attention window or not. Defaults to False.
        extra_norm (bool): Whether add extra norm at the end of main branch.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        attn_cfgs (dict): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        pretrained_window_size (int): Window size in pretrained.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=8,
                 shift=False,
                 extra_norm=False,
                 ffn_ratio=4.,
                 drop_path=0.,
                 pad_small_map=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained_window_size=0,
                 init_cfg=None):

        super(SwinBlockV2, self).__init__(init_cfg)
        self.with_cp = with_cp
        self.extra_norm = extra_norm

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': pad_small_map,
            **attn_cfgs
        }
        # use V2 attention implementation
        _attn_cfgs.update(
            window_msa=WindowMSAV2,
            pretrained_window_size=to_2tuple(pretrained_window_size))
        self.attn = ShiftWindowMSA(**_attn_cfgs)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            'add_identity': False,
            **ffn_cfgs
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        # add extra norm for every n blocks in huge and giant model
        if self.extra_norm:
            self.norm3 = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            # Use post normalization
            identity = x
            x = self.attn(x, hw_shape)
            x = self.norm1(x)
            x = x + identity

            identity = x
            x = self.ffn(x)
            x = self.norm2(x)
            x = x + identity

            if self.extra_norm:
                x = self.norm3(x)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockV2Sequence(BaseModule):
    """Module with successive Swin Transformer blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive swin transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        downsample (bool): Downsample the output of blocks by patch merging.
            Defaults to False.
        downsample_cfg (dict): The extra config of the patch merging layer.
            Defaults to empty dict.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        extra_norm_every_n_blocks (int): Add extra norm at the end of main
            branch every n blocks. Defaults to 0, which means no needs for
            extra norm layer.
        pretrained_window_size (int): Window size in pretrained.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=8,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 with_cp=False,
                 pad_small_map=False,
                 extra_norm_every_n_blocks=0,
                 pretrained_window_size=0,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        if downsample:
            self.out_channels = 2 * embed_dims
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': self.out_channels,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = PatchMerging(**_downsample_cfg)
        else:
            self.out_channels = embed_dims
            self.downsample = None

        self.blocks = ModuleList()
        for i in range(depth):
            extra_norm = True if extra_norm_every_n_blocks and \
                (i + 1) % extra_norm_every_n_blocks == 0 else False
            _block_cfg = {
                'embed_dims': self.out_channels,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'extra_norm': extra_norm,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'pretrained_window_size': pretrained_window_size,
                **block_cfgs[i]
            }
            block = SwinBlockV2(**_block_cfg)
            self.blocks.append(block)

    def forward(self, x, in_shape):
        if self.downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape

        for block in self.blocks:
            x = block(x, out_shape)

        return x, out_shape


@MODELS.register_module()
class wtSwinV2_wtm3d(BaseBackbone):
    """Swin Transformer V2.

    A PyTorch implement of : `Swin Transformer V2:
    Scaling Up Capacity and Resolution
    <https://arxiv.org/abs/2111.09883>`_

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        arch (str | dict): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (List[int]): The number of heads in attention
              modules of each stage.
            - **extra_norm_every_n_blocks** (int): Add extra norm at the end
              of main branch every n blocks.

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int | Sequence): The height and width of the window.
            Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        interpolate_mode (str): Select the interpolate mode for absolute
            position embeding vector resize. Defaults to "bicubic".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``
        stage_cfgs (Sequence[dict] | dict): Extra config dict for each
            stage. Defaults to an empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of
            each layer.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SwinTransformerV2
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'padding': 'same'}))
        >>> self = SwinTransformerV2(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32],
                         'extra_norm_every_n_blocks': 0}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48],
                         'extra_norm_every_n_blocks': 0}),
        # head count not certain for huge, and is employed for another
        # parallel study about self-supervised learning.
        **dict.fromkeys(['h', 'huge'],
                        {'embed_dims': 352,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [8, 16, 32, 64],
                         'extra_norm_every_n_blocks': 6}),
        **dict.fromkeys(['g', 'giant'],
                        {'embed_dims': 512,
                         'depths':     [2,  2, 42,  4],
                         'num_heads':  [16, 32, 64, 128],
                         'extra_norm_every_n_blocks': 6}),
    }  # yapf: disable

    _version = 1
    num_extra_tokens = 0

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
                 arch='tiny',
                 img_size=256,
                 patch_size=4,
                 in_channels=3,
                 window_size=8,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 out_indices=(0,1,2,3),
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 pretrained_window_sizes=[0, 0, 0, 0],
                 
                 init_cfg=None
                 ):
        super(wtSwinV2_wtm3d, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads',
                'extra_norm_every_n_blocks'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
        
        self.wtm_in_dim = wtm_in_dim
        self.wtm_dim = wtm_dim
        self.wtm_out_dim = wtm_out_dim
        self.wtm_kernel_size = wtm_kernel_size
        self.wtm_num_heads = wtm_num_heads
        self.wtm_dilations = wtm_dilations
        self.wtm_size = wtm_size
        self.wtm_low_level_dim = wtm_low_level_dim
        self.wtm_reduction = wtm_reduction
        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.extra_norm_every_n_blocks = self.arch_settings[
            'extra_norm_every_n_blocks']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

        if isinstance(window_size, int):
            self.window_sizes = [window_size for _ in range(self.num_layers)]
        elif isinstance(window_size, Sequence):
            assert len(window_size) == self.num_layers, \
                f'Length of window_sizes {len(window_size)} is not equal to '\
                f'length of stages {self.num_layers}.'
            self.window_sizes = window_size
        else:
            raise TypeError('window_size should be a Sequence or int.')

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        # self.patch_embed = PatchEmbed(**_patch_cfg)
        # self.patch_resolution = self.patch_embed.init_out_size

        # if self.use_abs_pos_embed:
        #     num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        #     self.absolute_pos_embed = nn.Parameter(
        #         torch.zeros(1, num_patches, self.embed_dims))
        #     self._register_load_state_dict_pre_hook(
        #         self._prepare_abs_pos_embed)

        self._register_load_state_dict_pre_hook(self._delete_reinit_params)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i > 0 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': self.window_sizes[i],
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'extra_norm_every_n_blocks': self.extra_norm_every_n_blocks,
                'pretrained_window_size': pretrained_window_sizes[i],
                'downsample_cfg': dict(use_post_norm=True),
                **stage_cfg
            }

            stage = SwinBlockV2Sequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, embed_dims[i + 1])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)
        self.resnet = ResNet()
        _wtm_cfg = {
            'in_dim': self.wtm_in_dim,
            'dim': self.wtm_dim,
            'out_dim': self.wtm_out_dim,
            'kernel_size': self.wtm_kernel_size,
            'num_heads': self.wtm_num_heads,
            'dilations': self.wtm_dilations,
            'size': self.wtm_size,
            'low_level_dim': self.wtm_low_level_dim,
            'reduction':self.wtm_reduction
        }
        self.wtm = WTM3D_A(**_wtm_cfg)
        self.conv3 = nn.Conv2d(256, self.embed_dims, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.embed_dims)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        super(wtSwinV2_wtm3d, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward(self, x):
        # x, hw_shape = self.patch_embed(x)
        resnet_out = self.resnet(x)
        hw_shape = resnet_out.shape[2], resnet_out.shape[3]
        x = self.conv3(resnet_out)
        x = self.bn3(x)
        x = self.relu(x)
        # print(hw_shape)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).permute(0,2,1)

        # if self.use_abs_pos_embed:
        #     x = x + resize_pos_embed(
        #         self.absolute_pos_embed, self.patch_resolution, hw_shape,
        #         self.interpolate_mode, self.num_extra_tokens)
        # x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1,
                                                           2).contiguous()
                outs.append(out)
        x = self.wtm(outs, resnet_out)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(wtSwinV2_wtm3d, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'absolute_pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.absolute_pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                'Resize the absolute_pos_embed shape from '
                f'{ckpt_pos_embed_shape} to {self.absolute_pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    def _delete_reinit_params(self, state_dict, prefix, *args, **kwargs):
        # delete relative_position_index since we always re-init it
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        logger.info(
            'Delete `relative_position_index` and `relative_coords_table` '
            'since we always re-init these params according to the '
            '`window_size`, which might cause unwanted but unworried '
            'warnings when loading checkpoint.')
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_position_index' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_coords_table' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]
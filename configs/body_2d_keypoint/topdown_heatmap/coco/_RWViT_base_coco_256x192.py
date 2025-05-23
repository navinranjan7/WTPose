_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(256, 192), heatmap_size=(64, 48), sigma=2)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResnetWViT',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        con_patch_size=(16,12),
        out_indices=(2,5,8,11),
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        con_in_channels=768,
        con_out_channels=128,
        con_ratio=3,
        ## WTM_A{
        wtm_in_dim=512,   # sum of all channels
        wtm_kernel_size = 7,
        wtm_dim=128, 
        wtm_out_dim=128,
        wtm_num_heads = [8, 8, 8, 8],
        wtm_dilations = [2,1,4,1,6,1,8,1],
        wtm_size=(64, 48),
        wtm_low_level_dim=256,
        wtm_reduction=32,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'v1/pretrained_models/mae_pretrain_vit_base_20230913.pth'),
    ),
    # backbone=dict(
    #     type='ViT',
    #     img_size=(256, 192),
    #     patch_size=16,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     ratio=1,
    #     use_checkpoint=False,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     drop_path_rate=0.3,
    # ),
    head=dict(
        type='HeatmapHeadWTPose',
        in_channels=128,
        out_channels=17,
        deconv_out_channels=False,
        conv_out_channels=False,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))
    # head=dict(
    #     type='HeatmapHead',
    #     in_channels=128,
    #     out_channels=17,
    #     deconv_out_channels=(128, 128),
    #     deconv_kernel_sizes=(4, 4),
    #     loss=dict(type='KeypointMSELoss', use_target_weight=True),
    #     decoder=codec),
    # test_cfg=dict(
    #     flip_test=True,
    #     flip_mode='heatmap',
    #     shift_heatmap=False,
    # ))

# base dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
data_mode = 'topdown'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=48,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # ann_file='annotations/person_keypoints_val2017.json',
        # bbox_file='coco/data/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

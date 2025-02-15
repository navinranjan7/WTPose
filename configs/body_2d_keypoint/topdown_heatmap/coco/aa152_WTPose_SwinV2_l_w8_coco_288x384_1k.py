_base_ = ['../../../_base_/default_runtime.py']
# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

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
auto_scale_lr = dict(base_batch_size=256)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=2)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

# pretrained = ('https://github.com/SwinTransformer/storage/releases/download'
              # '/v2.0.0/swinv2_base_patch4_window8_256.pth')
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='wtSwinV2',
        arch='large',
        drop_path_rate=0.3,
        ## WTM_A{
        wtm_in_dim=2880,   # sum of all channels
        wtm_kernel_size = 7,
        wtm_dim=192, 
        wtm_out_dim=192,
        wtm_num_heads = [8, 8, 8, 8],
        wtm_dilations = [2,1,4,1,6,1,8,1],
        wtm_size=(96, 72),
        wtm_low_level_dim=256,
        wtm_reduction=32,
        ## }WTM_A
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/nr4325/Desktop/Pose4/mmpose/ImageNet_pretrained_model/resnet152_swinv2_l.pth')
    ),
    # neck=dict(
    #     type='WTM_A',
    #     in_dim=1440,   # sum of all channels
    #     kernel_size = 7,
    #     dim=96, 
    #     out_dim=96,
    #     num_heads = [8, 8, 8, 8],
    #     dilations = [2,1,4,1,6,1,8,1],
    #     size=(64, 64),
    #     low_level_dim=256,
    #     reduction_ratio=8,
    # ),
    head=dict(
        type='HeatmapHeadWTPose',
        in_channels=192,
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

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=14,
    num_workers=2,
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
    batch_size=14,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
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

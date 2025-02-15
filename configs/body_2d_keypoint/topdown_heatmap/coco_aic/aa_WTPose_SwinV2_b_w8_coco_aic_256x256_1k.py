_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=1)

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
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# keypoint mappings
keypoint_mapping_coco = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
]

keypoint_mapping_aic = [
    (0, 6),
    (1, 8),
    (2, 10),
    (3, 5),
    (4, 7),
    (5, 9),
    (6, 12),
    (7, 14),
    (8, 16),
    (9, 11),
    (10, 13),
    (11, 15),
    (12, 17),
    (13, 18),
]

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
        arch='base',
        drop_path_rate=0.3,
        ## WTM_A{
        wtm_in_dim=1920,   # sum of all channels
        wtm_kernel_size = 7,
        wtm_dim=128, 
        wtm_out_dim=128,
        wtm_num_heads = [8, 8, 8, 8],
        wtm_dilations = [2,1,4,1,6,1,8,1],
        wtm_size=(64, 64),
        wtm_low_level_dim=256,
        wtm_reduction=32,
        ## }WTM_A
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/home/nr4325/Desktop/Pose4/mmpose/work_dirs/aa_WTPose_SwinV2_b_w8_coco_256x256_1k/best_coco_AP_epoch_200.pth')
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
        in_channels=128,
        out_channels=19,
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
# dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root_coco = 'data/coco/'
data_root_aic = 'data/aic/'

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

# train datasets
# dataset_coco = dict(
#     # type='RepeatDataset',
#     # dataset=dict(
#     type='CocoDataset',
#     data_root=data_root,
#     data_mode=data_mode,
#     ann_file='coco/annotations/person_keypoints_train2017.json',
#     data_prefix=dict(img='detection/coco/train2017/'),
#     pipeline=[
#         dict(
#             type='KeypointConverter',
#             num_keypoints=19,
#             mapping=keypoint_mapping_coco)
#     ],
#     )

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root_aic,
    data_mode=data_mode,
    ann_file='annotations/aic_train.json',
    data_prefix=dict(img='ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=19,
            mapping=keypoint_mapping_aic)
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_aic.py'),
        datasets=[dataset_aic],
        pipeline=train_pipeline,
        test_mode=False,
    ))
# train_dataloader = dict(
#     batch_size=32,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type='AicDataset',
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='aic/annotations/aic_train.json',
#         data_prefix=dict(img='ai_challenger_keypoint_train_20170902/'
#                          'keypoint_train_images_20170902/'),
#         pipeline=train_pipeline,
#     ))


val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root_coco,
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
    ann_file=data_root_coco + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
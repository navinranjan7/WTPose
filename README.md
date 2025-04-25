# **WTPose: Waterfall Transformer for Multi-Person Pose Estimaiton** 
[T4V@CVPR 2023 (Poster)](https://arxiv.org/abs/2411.18944) | [WACV Workshop 2025 (Oral)]([https://github.com/navinranjan7/WTPose/blob/main/WTPose%20Waterfall%20Transformer%20for%20Multi-Person%20Pose%20Estimation.pdf](https://openaccess.thecvf.com/content/WACV2025W/PRAW/html/Ranjan_WTPose_Waterfall_Transformer_for_Multi-person_Pose_Estimation_WACVW_2025_paper.html))


## **Description**
This repository contains the implementation and experiments related to our paper: **"WTPose: Waterfall Transformer for Pose Estimation"**. WTPose introduces a novel **Waterfall Transformer Module (WTM)** that enhances pose estimation by improving the performance of vision transformers like the Shifted Window (Swin) transformer. 

<p align="center">
  <img src="https://github.com/navinranjan7/WTPose/blob/main/resources/WTPose.png" title="Waterfall Transformer Framework for Multi-persion Pose Estimation.">
  Figure 1: Waterfall transformer framework for multi-person pose estimation. The input color image is fed through the modified Swin
Transformer backbone and WTM module to obtain 128 feature channels at reduced resolution by a factor of 4. The decoder module
generates K heatmaps, one per joint.
</p><br />

The WTM processes feature maps from multiple levels of the backbone through its waterfall branches. It applies filtering operations based on a **dilated attention mechanism** to expand the Field-of-View (FOV) and effectively capture both local and global context. These innovations lead to significant improvements over baseline models.

<p align="center">
  <img src="https://github.com/navinranjan7/WTPose/blob/main/resources/WTM.png" title="Waterfall Transformer Module">
  Figure 2: The proposed waterfall transformer module. The inputs are multi-scale feature maps from all four stages of the Swin backbone
and low-level features from the ResNet bottleneck. The waterfall module creates a waterfall flow, initially processing the input and then
creating a new branch. The feature dimensions (spatial and channel dimensions) output by various blocks are shown in parentheses.
</p><br />

Pose estimation examples using WTPose, showcasing the effectiveness of our approach.

<p align="center">
  <img src="https://github.com/navinranjan7/WTPose/blob/main/resources/WTPose_base_COCO_result.png" width="80%" title="Pose Estimation Result 1">
</p>
<p align="center">
  <img src="https://github.com/navinranjan7/WTPose/blob/main/resources/WTPose_examples.png" width="80%" title="Pose Estimation Result 2">
</p>

## Results on MS COCO Val set
Using detection results from a detector that obtains 56 mAP on person. The configs here are for both training and test.

> 256x192 resolution

|Model | Resolution | Params (M) | GFLOPs | AP | AR | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Swin-T | 256x192 | 32.8 | 6.3 | 72.44 | 78.20 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-t-p4-w7_8xb32-210e_coco-256x192.py) | log | - |
| WTPose-T | 256x192 | 30.0 | 12.8 | 74.23 | 79.43 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/aa_wtpose_swin_t_w7_coco_256x192_1k.py) | log | - |
| Swin-B | 256x192 | 93.0 | 19.0 | 73.72 | 79.32 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-b-p4-w7_8xb32-210e_coco-256x192.py) | log | - |
| WTPose-B | 256x192 | 89.3 | 25.6 | 74.96 | 80.51 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/wtm_resnet101_swin_b_w7_coco_256x192_1.py) | [log](https://github.com/navinranjan7/WTPose/blob/main/logs/log2/WTPose-b_19704347.out) | - |
| Swin-L | 256x192 | 32.8 | 41.0 | 74.30 | 79.82 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-l-p4-w7_8xb32-210e_coco-256x192.py) | log | - |
| WTPose-L | 256x192 | 32.8 | 47.9 | 75.40 | 80.81 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/aa_wtpose_swin-l-p4-w7_coco_256x192.py) | log | - |


> 384x288 resolution

|Model | Resolution | Params (M) | GFLOPs | AP | AR | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Swin-T | 384x288 | 32.8 | 13.8 | 74.89 | 80.93 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin_t-p4-w7_8xb32-210e_coco-384x288.py) | log | -  |
| WTPose-T | 384x288 | 30.0 | 28.3 | 76.36 | 81.43 | [config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/aa_wtm_ResNet101_swin_t_coco_384x288.py) | log | - |
| Swin-B | 384x288 | 93.0 | 41.6 | 75.81 | 80.99 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-b-p4-w7_8xb32-210e_coco-384x288.py) | log | - |
| WTPose-B | 384x288 | 89.3 | 55.8 | 77.18 | 82.07 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/aa_wtpose_swin_b_w7_coco_384x288.py) | log | - |
| Swin-L | 384x288 | 32.8 | 88.2 | 76.30 | 81.44 |[config](https://github.com/navinranjan7/WTPose/blob/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-l-p4-w7_8xb32-210e_coco-384x288.py) | log | - |
| WTPose-L | 384x288 | 32.8 | 104.2 | 77.56 | 82.61 |config | log | - |


# Installation and Setup

To use this code, clone the repository and refer to [installation.md](https://mmpose.readthedocs.io/en/latest/installation.html) for more detailed installation and dataset preparation. Alternatively, you can use [environment.yaml](https://github.com/navinranjan7/WTPose/blob/main/enviroment.yaml) to create an environment.

## **Usage**
Run the following command to start training or evaluation:
```bash
python tools\train.py --config
```

## **Citation**
If you use our work, please consider citing:
```bibtex
@article{ranjan2024waterfall,
  author    = {Ranjan, Navin and Artacho, Bruno and Savakis, Andreas},
  title     = {Waterfall Transformer for Multi-person Pose Estimation},
  journal   = {arXiv preprint arXiv:2411.18944},
  year      = {2024}
}
```
## Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmdetection).

## **Contact**
For any questions or discussions, feel free to open an issue or contact us at:
📧 nr4325@g.rit.edu

---
⭐ If you find this repository useful, please consider giving it a star!

# **WTPose: Waterfall Transformer for Multi-Person Pose Estimaiton**

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

## **Paper Link**
📄 [T4V@CVPR 2023 (Poster)](https://arxiv.org/abs/2411.18944)
📄 [WACV Workshop 2025 (Oral)](https://github.com/navinranjan7/WTPose/blob/main/WTPose%20Waterfall%20Transformer%20for%20Multi-Person%20Pose%20Estimation.pdf)

## **Installation**
To use this code, clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```

## **Usage**
Run the following command to start training or evaluation:
```bash
python main.py --config config.yaml
```

## **Images & Visualizations**
Below are some key visualizations from our research:

![Model Architecture](path/to/your-image.png)
*Figure 1: Model Architecture Overview.*

![Training Graph](path/to/another-image.png)
*Figure 2: Training Performance Comparison.*

## **Results**
Our approach achieves state-of-the-art results in [mention dataset or benchmark]. Key results include:

| Method | Accuracy (%) | Model Size (MB) | Inference Speed (ms) |
|--------|-------------|----------------|----------------------|
| Baseline | XX.X | XX | XX |
| Proposed | **XX.X** | **XX** | **XX** |

For more details, check the **Results** section in our paper.

## **Citation**
If you use our work, please consider citing:
```bibtex
@article{yourpaper2025,
  author    = {Your Name and Co-authors},
  title     = {Your Paper Title},
  journal   = {Journal/Conference Name},
  year      = {2025},
  doi       = {your-doi-here}
}
```

## **Contact**
For any questions or discussions, feel free to open an issue or contact us at:
📧 your.email@example.com

---
⭐ If you find this repository useful, please consider giving it a star!

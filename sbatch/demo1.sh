#!/bin/bash -l

#SBATCH --job-name=demo		# Name for your job
#SBATCH --comment="work"		# Comment for your job

#SBATCH --account=segmesh		# Project account to run your job under
#SBATCH --partition=tier3	# Partition to run your job on

#SBATCH --output=%x_%j.out		# Output file
#SBATCH --error=%x_%j.err		# Error file

#SBATCH --mail-user=nr4325@rit.edu	# Email address to notify
#SBATCH --mail-type=END			# Type of notification emails to send

#SBATCH --time=0-01:00:00		# Time limit
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=2			# How many tasks per node
#SBATCH --cpus-per-task=2		# Number of CPUs per task
#SBATCH --mem-per-cpu=30g		# Memory per CPU

#SBATCH --gres=gpu:a100:2
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate pose4

# image_path = '/home/nr4325/Desktop/Pose4/mmpose/data/coco/val2017/'
# out_img_root = '/home/nr4325/Desktop/Pose4/results/'
# command_string = 'python demo/topdown_demo_with_mmdet.py ' + \
#     'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py ' + \
#     'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ' + \
#     'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py ' + \
#     'weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth ' + \
#     '--input ' + image_path + ' --output-root ' + out_img_root 

# python demo/topdown_demo_with_mmdet.py \
#     demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
#     https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth  \
#     /home/nr4325/Desktop/Pose4/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
#     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
#     --input '/home/nr4325/Desktop/Pose4/mmpose/data/coco/val2017/000000000139.jpg'  \
#     --output-root wtm/
python demo/image_demo.py \
    /home/nr4325/Desktop/Pose4/mmpose/tests/data/coco/000000197388.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file wtm/result1.jpg \
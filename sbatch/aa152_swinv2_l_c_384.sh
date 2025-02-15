#!/bin/bash -l

#SBATCH --job-name=S2l3		# Name for your job
#SBATCH --comment="work"		# Comment for your job

#SBATCH --account=segmesh		# Project account to run your job under
#SBATCH --partition=tier3	# Partition to run your job on

#SBATCH --output=%x_%j.out		# Output file
#SBATCH --error=%x_%j.err		# Error file

#SBATCH --mail-user=nr4325@rit.edu	# Email address to notify
#SBATCH --mail-type=END			# Type of notification emails to send

#SBATCH --time=5-00:00:00		# Time limit
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=4			# How many tasks per node
#SBATCH --cpus-per-task=2		# Number of CPUs per task
#SBATCH --mem-per-cpu=30g		# Memory per CPU

#SBATCH --gres=gpu:a100:4
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate pose4

python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=4 \
    tools/train.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/aa152_WTPose_SwinV2_l_w8_coco_288x384_1k.py \
    --launcher pytorch ${@:3}

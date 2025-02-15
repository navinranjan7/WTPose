#!/bin/bash -l

#SBATCH --job-name=swin_pose		# Name for your job
#SBATCH --comment="work"		# Comment for your job

#SBATCH --account=segmesh		# Project account to run your job under
#SBATCH --partition=debug	# Partition to run your job on

#SBATCH --output=%x_%j.out		# Output file
#SBATCH --error=%x_%j.err		# Error file

#SBATCH --mail-user=nr4325@rit.edu	# Email address to notify
#SBATCH --mail-type=END			# Type of notification emails to send

#SBATCH --time=1-00:00:00		# Time limit
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=2			# How many tasks per node
#SBATCH --cpus-per-task=2		# Number of CPUs per task
#SBATCH --mem-per-cpu=30g		# Memory per CPU

#SBATCH --gres=gpu:a100:2
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate pose4

python -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=2 \
    tools/train.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py \
    --launcher pytorch ${@:3}

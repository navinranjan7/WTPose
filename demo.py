import os

image_path = '/home/nr4325/Desktop/Pose4/mmpose/data/coco/val2017/'
out_img_root = '/home/nr4325/Desktop/Pose4/results/'
command_string = 'python demo/topdown_demo_with_mmdet.py ' + \
    'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py ' + \
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ' + \
    'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py ' + \
    'weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth ' + \
    '--input ' + image_path + ' --output-root ' + out_img_root 

os.system(command_string)
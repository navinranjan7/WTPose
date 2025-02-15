import torch

swin = torch.load('/home/nr4325/Desktop/Pose4/mmpose/nav/pretrain/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth')
resnet101 = torch.load('/home/nr4325/Desktop/Pose4/mmpose/nav/pretrain/resnet101_8xb32_in1k_20210831-539c63f8.pth')

swin_checkpoint = swin
# print(swin_checkpoint.keys())
resnet_checkpoint = resnet101['state_dict']
# print(resnet_checkpoint.keys())

resnet_bottleneck = {key: value for key, value in resnet_checkpoint.items() if key.startswith('backbone.layer1.')}
# print(resnet_bottleneck.keys())

new_state_dict = {}
for key, value in resnet_bottleneck.items():
    # Start with the original key
    new_key = key

    # Apply renaming rules sequentially
    if "backbone." in new_key:
        new_key = new_key.replace("backbone.", "resnet.")
    
    new_state_dict[new_key] = value

# print("Updated keys:", new_state_dict.keys())

merged_state_dict = {**new_state_dict, **swin_checkpoint}
# print(merged_state_dict)
# # Update the checkpoint with the modified state_dict
resnet101_swin = {}
resnet101_swin = merged_state_dict

print(merged_state_dict.keys())
# # Save the update.keysd weights
new_checkpoint_path = "/home/nr4325/Desktop/Pose4/mmpose/nav/pretrain/resnet101_swin_base_patch4_window12_384_22kto1k.pth"
torch.save(resnet101_swin, new_checkpoint_path)

# resnet_swin = torch.load('/home/nr4325/Desktop/Pose4/mmpose/nav/pretrain/resnet101_swin_base_patch4_window7_224_22k.pth')
# print(resnet_swin['state_dict'].keys())
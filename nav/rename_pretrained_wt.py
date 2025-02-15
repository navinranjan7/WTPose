import torch

# Load the .pth file
checkpoint_path = '/home/nr4325/Desktop/Pose4/mmpose/nav/pretrain/resnet101_swin_large_patch4_window7_224_22kto1k.pth'  # Path to your .pth file
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Access the state_dict
# state_dict = checkpoint['state_dict']  # Adjust the key based on your checkpoint structure
# print("Original keys:", state_dict.keys())
print(checkpoint.keys())
# Rename specific keys
# new_state_dict = {}
# for key, value in state_dict.items():
#     # Start with the original key
#     new_key = key

#     # Apply renaming rules sequentially
#     if "backbone." in new_key:
#         new_key = new_key.replace("backbone.", "")
# #     if "blocks." in new_key:
# #         new_key = new_key.replace("blocks.", "layers.")
# #     # if "proj." in new_key:
# #     #     new_key = new_key.replace("proj.", "projection.")
# #     if "norm1." in new_key:
# #         new_key = new_key.replace("norm1.", "ln1.")
# #     if "norm2." in new_key:
# #         new_key = new_key.replace("norm2.", "ln2.")
# #     if "mlp.fc1." in new_key:
# #         new_key = new_key.replace("mlp.fc1.", "ffn.layers.0.0.")
# #     if "mlp.fc2." in new_key:
# #         new_key = new_key.replace("mlp.fc2.", "ffn.layers.1.")
# #     if "patch_embed.proj." in new_key:
# #         new_key=new_key.replace("patch_embed.proj.","patch_embed.projection.")

# #     # Add the updated key-value pair to the new state_dict
#     new_state_dict[new_key] = value

# print("Updated keys:", new_state_dict.keys())

# # # Update the checkpoint with the modified state_dict
# checkpoint['state_dict'] = state_dict

# # # Save the updated weights
# new_checkpoint_path = "/home/nr4325/Desktop/_swin_base_patch4_window7_224_22k.pth"
# torch.save(checkpoint, new_checkpoint_path)

# print(f"Updated checkpoint saved to {new_checkpoint_path}")


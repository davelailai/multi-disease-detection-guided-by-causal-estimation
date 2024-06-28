from mmpretrain import get_model
import torch
import pickle
import os
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model = get_model("configs/FFA/FFA_baseline.py")

# retifund=torch.load('RETFound_cfp_weights.pth')
# checkpoint_model = retifund['model']
# state_dict =model.backbone.state_dict()

   

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        # num_patches = model.patch_embed.num_patches
        patch_resolution = model.patch_embed.init_out_size
        num_patches = patch_resolution[0] * patch_resolution[1]
        num_extra_tokens = model.num_extra_tokens
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def convert_retifund(ckpt,mae):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        # new_k=k
        # print(k)
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.layers.head.')
            new_ckpt[new_k] = new_v
            continue
        elif k.startswith('patch_embed'):
            if 'proj.' in k:
                new_k = k.replace('proj.', 'projection.')
            else:
                new_k = k
        elif k.startswith('norm_pre'):
            new_k = k.replace('norm_pre', 'pre_norm')
    
        elif k.startswith('blocks'):
            new_k = k.replace('blocks.', 'layers.')
            if 'norm1' in k:
                new_k = new_k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = new_k.replace('norm2', 'ln2')
            elif 'mlp.fc1' in k:
                new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
        elif k.startswith('norm'):
            new_k = k.replace('norm', 'ln1')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k
        # print(new_k)
        new_ckpt[new_k] = new_v
    return new_ckpt

# for k in ['head.weight', 'head.bias']:
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]

retifund = CheckpointLoader.load_checkpoint('RETFound_cfp_weights.pth', map_location='cpu')

mae= CheckpointLoader.load_checkpoint('https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth',map_location='cpu')
checkpoint=retifund['model']

model = get_model("configs/FFA/FFA_baseline.py")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

weight = convert_retifund(state_dict,mae)

torch.save(weight, '/users/lailai/sharedscratch/openmmlab/mmpretrain/mmpretrain_RETFound_cfp_weights.pth')

print('Done!!')

# interpolate_pos_embed(model.backbone, checkpoint_model)

# import hashlib
# import os.path as osp
# from mmengine.runner import save_checkpoint
# model.backbone.load_state_dict(checkpoint_model, strict=False)

# checkpoint_data = pickle.dumps(model)
# sha = hashlib.sha256(checkpoint_data).hexdigest()
# final_path = '/users/lailai/sharedscratch/openmmlab/mmpretrain/mmpretrain_RETFound_cfp_weights.pth'
# torch.save(model, final_path)
# save_checkpoint(model, final_path)

# # Copyright (c) OpenMMLab. All rights reserved.
# import argparse
# import os.path as osp
# from collections import OrderedDict

# import mmengine
# import torch
# from mmengine.runner import CheckpointLoader


# def convert_retifund(ckpt):
#     new_ckpt = OrderedDict()

#     for k, v in list(ckpt.items()):
#         new_v = v
#         # if k.startswith('head'):
#         #     new_k = k.replace('head.', 'head.layers.head.')
#         #     new_ckpt[new_k] = new_v
#         #     continue
#         # elif k.startswith('patch_embed'):
#         #     if 'proj.' in k:
#         #         new_k = k.replace('proj.', 'projection.')
#         #     else:
#         #         new_k = k
#         # elif k.startswith('norm_pre'):
#         #     new_k = k.replace('norm_pre', 'pre_norm')
#         # elif k.startswith('blocks'):
#         #     new_k = k.replace('blocks.', 'layers.')
#         #     if 'norm1' in k:
#         #         new_k = new_k.replace('norm1', 'ln1')
#         #     elif 'norm2' in k:
#         #         new_k = new_k.replace('norm2', 'ln2')
#         #     elif 'mlp.fc1' in k:
#         #         new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0')
#         #     elif 'mlp.fc2' in k:
#         #         new_k = new_k.replace('mlp.fc2', 'ffn.layers.1')
#         # elif k.startswith('norm'):
#         #     new_k = k.replace('norm', 'ln1')
#         # else:
#         #     new_k = k

#         if not new_k.startswith('head'):
#             new_k = 'backbone.' + new_k
#         new_ckpt[new_k] = new_v
#     return new_ckpt


# def main():
#     parser = argparse.ArgumentParser(
#         description='Convert keys in pretrained clip '
#         'models to mmpretrain style.')
#     parser.add_argument('src', default='RETFound_cfp_weights.pth',help='src model path or url')
#     # The dst path must be a full path of the new checkpoint.
#     parser.add_argument('dst', default='MMpretrain_RETFound_cfp_weights.pth', help='save path')
#     args = parser.parse_args()

#     checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     else:
#         state_dict = checkpoint

#     weight = convert_retifund(state_dict)
#     mmengine.mkdir_or_exist(osp.dirname(args.dst))
#     torch.save(weight, args.dst)

#     print('Done!!')


# if __name__ == '__main__':
#     main()





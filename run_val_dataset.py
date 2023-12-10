import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import torch.nn as nn

from PIL import Image

from pathlib import Path

from timm.models import create_model

import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE, build_pretraining_dataset

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Agg")
torch.manual_seed(2023)

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--data_path', default="../archive/masked_crop/val", type=str, help='dataset path')
    parser.add_argument('--save_path', default="./output", type=str, help='save image path')
    parser.add_argument('--model_path', default="./checkpoint/masked_40.pth", type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def show(ori, mask, rec, args):
    def show_gray(image, title=''):
        plt.imshow(image, cmap='Greys_r')
        plt.title(title, fontsize=16)
        plt.axis("off")
        return
    
    plt.rcParams['figure.figsize'] = [10, 10]
    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['savefig.dpi'] = 200
    B = ori.shape[0]

    for i in range(B):
        ori_title = ""
        mask_title = ""
        rec_title = ""
        if i == 0:
            ori_title = "origin"
            mask_title = "mask"
            rec_title = "reconstruction"
        plt.subplot(3, B, i+1)
        show_gray(ori[i], ori_title)
        plt.subplot(3, B, i+B+1)
        show_gray(mask[i], mask_title)
        plt.subplot(3, B, i+2*B+1)
        show_gray(rec[i], rec_title)
    
    save_name = os.path.join(args.save_path, "val_result.png")

    plt.savefig(fname=save_name, dpi=400)

    # plt.show()


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    dataset_val = build_pretraining_dataset(args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=dataset_val.__len__()
    )

    loss_func = nn.MSELoss()

    with torch.no_grad():
        for idx, (batch, _) in enumerate(data_loader_val):
            images, bool_masked_pos = batch
            img = images.to(device, non_blocking=True).float()
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[0]).to(device, non_blocking=True)
            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

            outputs = model(img, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

            loss_value = loss.item()
            print("Val loss is {}".format(loss_value))

            #save original img
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            ori_img = img * std + mean  # in [0, 1], and shape is [batch_size, c, h, w]
            # img = ToPILImage()(ori_img[0, :])
            # img.save(f"{args.save_path}/ori_img.jpg")

            img_squeeze = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
            # img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            img_patch = rearrange(img_squeeze, 'b n p c -> b n (p c)')
            # img_patch[bool_masked_pos] = outputs
            img_patch[bool_masked_pos] = rearrange(outputs, 'b n c -> (b n) c')

            #make mask
            mask = torch.ones_like(img_patch)
            mask[bool_masked_pos] = 0
            mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
            mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

            #save reconstruction img
            rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
            # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
            # rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
            rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)
            rec_img = rec_img * std + mean
            # img = ToPILImage()(rec_img[0, :].clip(0,0.996))
            # img.save(f"{args.save_path}/rec_img.jpg")

            #save random mask img
            img_mask = rec_img * mask
            # img = ToPILImage()(img_mask[0, :])
            # img.save(f"{args.save_path}/mask_img.jpg")
            ori_img = rearrange(ori_img, 'b c h w -> b h w c').cpu()
            img_mask = rearrange(img_mask, 'b c h w -> b h w c').cpu()
            rec_img = rearrange(rec_img, 'b c h w -> b h w c').cpu()
            show(ori_img, img_mask, rec_img, args)





if __name__ == '__main__':
    opts = get_args()
    opts.data_path = "../archive/masked_crop/val"
    opts.save_path = "./output"
    opts.model_path = "./checkpoint/masked_40.pth"
    main(opts)

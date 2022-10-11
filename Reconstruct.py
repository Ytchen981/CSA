import os
import torch
# import numpy as np
# import random
import xlwt
import math
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from pathlib import Path
import math
import apex.amp as amp
from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

from utils.parse_arg import parse_args, cfg
from utils.utils import set_seed, DupStdoutFileManager, print_easydict, transform_train, transform_test, transform_train_norm, transform_test_norm, cifar10_std, cifar10_mean
from utils.Shapley_sample import getShapley_pixel, getShapley_freq
from utils.Shapley_visual import visual_shap, visual_shap_w_tick
from utils.frequency import transform_ifft, transform_fft

parse_args('Reconstruct images.')

from model.resnet import *


def reconstruct(path):
    recon_path = os.path.join(path, "reconstruction")
    ifft_path = os.path.join(path, "ifft")
    shap_path = os.path.join(path, "shap_result")
    if not Path(recon_path).exists():
        Path(recon_path).mkdir()
    if not Path(ifft_path).exists():
        Path(ifft_path).mkdir()
    '''min = 10000
    for i in range(10):
        result_path = os.path.join(shap_path, f"{i}")
        files = os.listdir(result_path)
        num = 0
        for file in files:
            if ".png" in file and not "shap" in file:
                num += 1
        if num < min:
            min = num'''
    classes = os.listdir(shap_path)
    for i in classes:
        result_path = os.path.join(shap_path, i)
        if os.path.isdir(result_path):
            files = os.listdir(result_path)
            # num = 0
            for file in files:
                if ".png" in file and not "shap" in file:
                    # num += 1
                    # if num > min:
                    # break
                    with open(os.path.join(result_path, file), 'rb') as f:
                        img = Image.open(f)
                        img = img.convert('RGB')
                    if cfg.train.norm:
                        img = transform_test_norm(img).unsqueeze(0)
                    else:
                        img = transform_test(img).unsqueeze(0)
                    freq = transform_fft(img)
                    freq_shap_file = file.split('.')[0] + "_freq.pt"
                    freq_shap = torch.load(os.path.join(result_path, freq_shap_file))
                    mask_size = int(math.sqrt(freq_shap.numel()))
                    freq_shap = freq_shap.view(1, 1, mask_size, mask_size)
                    freq_shap = freq_shap.expand(1, 3, mask_size, mask_size)
                    freq_shap = F.interpolate(freq_shap, size=[img.size(2), img.size(3)], mode="nearest").float()
                    mask = (freq_shap > 0).int()
                    mask[:, :, img.size(2) // 2, img.size(3) // 2] = 0
                    pos_freq = freq * mask
                    mask = (freq_shap < 0).int()
                    mask[:, :, img.size(2) // 2, img.size(3) // 2] = 0
                    neg_freq = freq * mask
                    pos_img = transform_ifft(pos_freq)
                    # print(f"max:{torch.max(pos_img)}, min:{torch.min(pos_img)}")
                    # pos_img = (pos_img + torch.min(pos_img)) / (torch.max(pos_img) - torch.min(pos_img))
                    neg_img = transform_ifft(neg_freq)
                    # neg_img = (neg_img + torch.min(neg_img)) / (torch.max(neg_img) - torch.min(neg_img))
                    # print(f"max:{torch.max(neg_img)}, min:{torch.min(neg_img)}")
                    recon_class_path = os.path.join(recon_path, f"{i}")
                    if not Path(recon_class_path).exists():
                        Path(recon_class_path).mkdir()
                    ifft_class_path = os.path.join(ifft_path, i)
                    if not Path(ifft_class_path).exists():
                        Path(ifft_class_path).mkdir()
                    torch.save(pos_img, os.path.join(ifft_path, i, file.split('.')[0] + "_pos.pt"))
                    torch.save(neg_img, os.path.join(ifft_path, i, file.split('.')[0] + "_neg.pt"))

                    max_shap = torch.max(freq_shap).item()
                    min_shap = torch.min(freq_shap).item()
                    for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                        mask = (freq_shap > (threshold * max_shap)).int()
                        mask[:, :, img.size(2) // 2, img.size(3) // 2] = 1
                        pos_freq = freq * mask
                        mask = (freq_shap < (threshold * min_shap)).int()
                        mask[:, :, img.size(2) // 2, img.size(3) // 2] = 1
                        neg_freq = freq * mask
                        pos_img = transform_ifft(pos_freq)
                        # print(f"max:{torch.max(pos_img)}, min:{torch.min(pos_img)}")
                        # pos_img = (pos_img + torch.min(pos_img)) / (torch.max(pos_img) - torch.min(pos_img))
                        neg_img = transform_ifft(neg_freq)
                        # neg_img = (neg_img + torch.min(neg_img)) / (torch.max(neg_img) - torch.min(neg_img))
                        # print(f"max:{torch.max(neg_img)}, min:{torch.min(neg_img)}")
                        pos_img = pos_img.squeeze()
                        neg_img = neg_img.squeeze()
                        if cfg.train.norm:
                            for j in range(3):
                                pos_img[j, :, :] = pos_img.squeeze()[j, :, :] * cifar10_std[j]
                                pos_img[j, :, :] = pos_img[j, :, :] + cifar10_mean[j]
                                neg_img[j, :, :] = neg_img.squeeze()[j, :, :] * cifar10_std[j]
                                neg_img[j, :, :] = neg_img[j, :, :] + cifar10_mean[j]
                        save_image(pos_img,
                                   os.path.join(recon_class_path, file.split('.')[0] + f"_pos_{threshold}.png"))
                        save_image(neg_img,
                                   os.path.join(recon_class_path, file.split('.')[0] + f"_neg_{threshold}.png"))

                    img = img.squeeze()
                    if cfg.train.norm:
                        for j in range(3):
                            img[j, :, :] = img[j, :, :] * cifar10_std[j]
                            img[j, :, :] = img[j, :, :] + cifar10_mean[j]
                    save_image(img, os.path.join(recon_class_path, file))

def generate_low_pass_mask(w, h, r):
    mask = torch.ones(1, 3, w, h)
    maximum = math.sqrt((w/2)**2 + (h/2)**2)
    for j in range(w):
        for k in range(h):
            if math.sqrt((j - w/2)**2 + (k - h/2)**2) / maximum > r:
                mask[:, :, j, k] = 0
    return mask

def reconstruct_low_pass(path, r):
    recon_path = os.path.join(path, f"reconstruction_low{r}")
    shap_path = os.path.join(path, "shap_result")
    if not Path(recon_path).exists():
        Path(recon_path).mkdir()
    for i in range(10):
        result_path = os.path.join(shap_path, f"{i}")
        files = os.listdir(result_path)
        for file in files:
            if ".png" in file and not "shap" in file:
                with open(os.path.join(result_path, file), 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                img = transform_test(img).unsqueeze(0)
                low_pass_mask = generate_low_pass_mask(img.size(2), img.size(3), r)
                freq = transform_fft(img)
                freq_shap_file = file.split('.')[0] + "_freq.pt"
                freq_shap = torch.load(os.path.join(result_path, freq_shap_file))
                freq_shap = freq_shap.view(1, 1, cfg.shapley.mask_size, cfg.shapley.mask_size)
                freq_shap = freq_shap.expand(1, 3, cfg.shapley.mask_size, cfg.shapley.mask_size)
                freq_shap = F.interpolate(freq_shap, size=[img.size(2), img.size(3)],mode="nearest").float()
                mask = (freq_shap>0).int()
                mask[:,:, img.size(2) // 2, img.size(3) // 2] = 1
                pos_freq = freq * mask * low_pass_mask
                mask = (freq_shap<0).int()
                mask[:,:, img.size(2) // 2, img.size(3) // 2] = 1
                neg_freq = freq * mask * low_pass_mask
                pos_img = transform_ifft(pos_freq)
                neg_img = transform_ifft(neg_freq)
                recon_class_path = os.path.join(recon_path, f"{i}")
                if not Path(recon_class_path).exists():
                    Path(recon_class_path).mkdir()
                save_image(pos_img, os.path.join(recon_class_path,file.split('.')[0] + "_pos.png"))
                save_image(neg_img, os.path.join(recon_class_path,file.split('.')[0] + "_neg.png"))
                save_image(img, os.path.join(recon_class_path,file))

def generate_high_pass_mask(w, h, r):
    mask = torch.ones(1, 3, w, h)
    maximum = math.sqrt((w/2)**2 + (h/2)**2)
    for j in range(w):
        for k in range(h):
            if math.sqrt((j - w/2)**2 + (k - h/2)**2) / maximum < r:
                mask[:, :, j, k] = 0
    mask[:, :, w // 2, h // 2] = 1
    return mask

def reconstruct_high_pass(path, r):
    recon_path = os.path.join(path, f"reconstruction_high{r}")
    shap_path = os.path.join(path, "shap_result")
    if not Path(recon_path).exists():
        Path(recon_path).mkdir()
    for i in range(10):
        result_path = os.path.join(shap_path, f"{i}")
        files = os.listdir(result_path)
        for file in files:
            if ".png" in file and not "shap" in file:
                with open(os.path.join(result_path, file), 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                img = transform_test(img).unsqueeze(0)
                high_pass_mask = generate_high_pass_mask(img.size(2), img.size(3), r)
                freq = transform_fft(img)
                freq_shap_file = file.split('.')[0] + "_freq.pt"
                freq_shap = torch.load(os.path.join(result_path, freq_shap_file))
                freq_shap = freq_shap.view(1, 1, cfg.shapley.mask_size, cfg.shapley.mask_size)
                freq_shap = freq_shap.expand(1, 3, cfg.shapley.mask_size, cfg.shapley.mask_size)
                freq_shap = F.interpolate(freq_shap, size=[img.size(2), img.size(3)],mode="nearest").float()
                mask = (freq_shap>0).int()
                #mask[:,:, img.size(2) // 2, img.size(3) // 2] = 1
                pos_freq = freq * mask * high_pass_mask
                mask = (freq_shap<0).int()
                #mask[:,:, img.size(2) // 2, img.size(3) // 2] = 1
                neg_freq = freq * mask * high_pass_mask
                pos_img = transform_ifft(pos_freq)
                neg_img = transform_ifft(neg_freq)
                recon_class_path = os.path.join(recon_path, f"{i}")
                if not Path(recon_class_path).exists():
                    Path(recon_class_path).mkdir()
                save_image(pos_img, os.path.join(recon_class_path,file.split('.')[0] + "_pos.png"))
                save_image(neg_img, os.path.join(recon_class_path,file.split('.')[0] + "_neg.png"))
                save_image(img, os.path.join(recon_class_path,file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--shap_path', default=None, type=str)
    args = parser.parse_args()
    reconstruct(args.shap_path)
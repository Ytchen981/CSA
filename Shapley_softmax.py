import os
import torch
# import numpy as np
# import random
import xlwt
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from pathlib import Path
import apex.amp as amp
from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from Adv_methods.madrys import pgd

from utils.parse_arg import parse_args, cfg
from utils.utils import set_seed, DupStdoutFileManager, print_easydict, transform_train, transform_test, \
    transform_train_norm, transform_test_norm,cifar10_std, cifar10_mean, ImageNet_train_transform, ImageNet_test_transform
from utils.Shapley_sample import getShapley_pixel, getShapley_freq, getShapley_freq_dis, sample_mask, getShapley_freq_softmax
from utils.Shapley_visual import visual_shap

parse_args('Training model.')

#from model.resnet import *
from model.vgg import *
from model.alexnet import *

def main(model, dataloader, nclass):
    count = [0 for _ in range(nclass)]
    model.eval()
    model.cuda()

    for i, (img, y) in enumerate(dataloader):
        bs, c, w, h = img.size()
        img = img.cuda()
        y = y.cuda()
        if cfg.shapley.adv_sample:
            img = pgd(model, img, y, step_size= 2 / 255, epsilon= 8 / 255, perturb_steps=20)
        for k in range(bs):
            flag = True
            if count[y[k].item()] < cfg.shapley.start_num:
                count[y[k].item()] += 1
            elif count[y[k].item()] < cfg.shapley.num_per_class:
                if Path(os.path.join(cfg.resume_path, "shap_result", f"{y[k].item()}")).exists():
                    resume_class_path = os.path.join(cfg.resume_path, "shap_result", f"{y[k].item()}")
                    if Path(os.path.join(resume_class_path, f"{count[y[k].item()]}_freq.pt")).exists():
                        count[y[k].item()] += 1
                        flag = False
                if flag:
                    print(f"start sample for class {y[k].item()} num {count[y[k].item()]}")
                    tick = time.time()
                    if cfg.shapley.get_freq_by_dis:
                        shap_value = getShapley_freq_dis(img, y, model, cfg.shapley.sample_times, cfg.shapley.mask_size,
                                                         k)
                    else:
                        if cfg.shapley.fix_mask:
                            shap_value = getShapley_freq_softmax(img, y, model, cfg.shapley.sample_times, cfg.shapley.mask_size,
                                                         k, n_per_batch=cfg.shapley.n_per_batch,
                                                         split_n=cfg.shapley.split_n,
                                                         static_center=cfg.shapley.static_center, fix_masks=True,
                                                         mask_path=cfg.output_path)
                        else:
                            shap_value = getShapley_freq_softmax(img, y, model, cfg.shapley.sample_times, cfg.shapley.mask_size,
                                                         k, n_per_batch=cfg.shapley.n_per_batch,
                                                         split_n=cfg.shapley.split_n,
                                                         static_center=cfg.shapley.static_center)

                    shap_path = os.path.join(cfg.output_path, "shap_result", f"{y[k].item()}")
                    if not Path(shap_path).exists():
                        Path(shap_path).mkdir(parents=True)
                    shap_path = os.path.join(shap_path, f"{count[y[k].item()]}_freq.pt")
                    torch.save(shap_value, shap_path)
                    visual_path = os.path.join(cfg.output_path, "shap_result", f"{y[k].item()}",
                                               f"{count[y[k].item()]}_freq_shap.png")
                    visual_shap(shap_value, cfg.shapley.mask_size, cfg.shapley.mask_size, visual_path)

                    '''shap_value_pixel = getShapley_pixel(img, y, model, cfg.shapley.sample_times, cfg.shapley.mask_size, k)
                    shap_path = os.path.join(cfg.output_path, "shap_result", f"{y[k].item()}")
                    if not Path(shap_path).exists():
                        Path(shap_path).mkdir(parents=True)
                    shap_path = os.path.join(shap_path, f"{count[y[k].item()]}_pixel.pt")
                    torch.save(shap_value_pixel, shap_path)
                    visual_path = os.path.join(cfg.output_path, "shap_result", f"{y[k].item()}",
                                               f"{count[y[k].item()]}_pixel_shap.png")
                    visual_shap(shap_value_pixel, cfg.shapley.mask_size, cfg.shapley.mask_size, visual_path)
                    '''
                    raw_img = img[k].clone().cpu()
                    img_path = os.path.join(cfg.output_path, "shap_result", f"{y[k].item()}",
                                            f"{count[y[k].item()]}.png")
                    if cfg.train.norm:
                        for j in range(3):
                            raw_img[j, :, :] = raw_img[j, :, :] * cifar10_std[j]
                            raw_img[j, :, :] = raw_img[j, :, :] + cifar10_mean[j]
                    save_image(raw_img, img_path)
                    count[y[k].item()] += 1
                    print(
                        f"class {y[k].item()} num {count[y[k].item()]} finished time used:{time.time() - tick : .2f}s")
            else:
                pass


if __name__ == '__main__':
    set_seed(cfg.train.seed)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #tfboardwriter = SummaryWriter(logdir=str(Path(cfg.output_path) / 'tensorboard' / 'training_{}'.format(now_time)))

    if cfg.dataset == "cifar10":
        nclass=10
        if cfg.train.norm:
            trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train_norm)
            testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test_norm)
        else:
            trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test)
    elif cfg.dataset == "svhn":
        nclass = 10
        trainset = datasets.SVHN(root=cfg.data_path, split='train', download=True, transform=transform_train)
        testset = datasets.SVHN(root=cfg.data_path, split='test', download=True, transform=transform_test)
    elif cfg.dataset == "cifar100":
        nclass = 100
        trainset = datasets.CIFAR100(root=cfg.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=cfg.data_path, train=False, download=True, transform=transform_test)
    elif cfg.dataset == "imagenet":
        nclass = 1000
        trainset = datasets.ImageFolder(root=os.path.join(cfg.data_path, "train"),transform=ImageNet_train_transform)
        testset = datasets.ImageFolder(root=os.path.join(cfg.data_path, "val"), transform=ImageNet_test_transform)
    else:
        raise KeyError(f"Unknown model {cfg.dataset}")
    train_loader = DataLoader(trainset, batch_size=cfg.train.batchsize, shuffle=False, pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.num_workers)
    test_loader = DataLoader(testset, batch_size=cfg.train.batchsize, shuffle=False, pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.num_workers)

    assert nclass == 10
    if cfg.model == "ResNet18":
        if nclass == 200 or nclass == 1000:
            from torchvision import models as tv_model
            model = tv_model.resnet18(pretrained=False, num_classes=nclass)
        else:
            from model.resnet import *
            model = ResNet18(num_classes=nclass)
    elif "WRN" in cfg.model:
        model_name_list = cfg.model.split('-')
        depth = int(model_name_list[1])
        width = int(model_name_list[2])
        model = WideResNet(depth=depth, num_classes=nclass, widen_factor=width)
    elif cfg.model == "vgg16_bn":
        model = vgg16_bn(num_classes=nclass)
    elif cfg.model == "AlexNet":
        model = AlexNet(num_classes=nclass)
    else:
        raise KeyError(f"Unknown model {cfg.model}")

    if "robustbench" not in cfg.model:
        if "conf" in cfg.shapley.model_path:
            new_dict = {}
            checkpoint = torch.load(cfg.shapley.model_path, map_location='cpu')
            flag = False
            for key in checkpoint.keys():
                if "backbone" in key:
                    flag = True
                    break
            if flag:
                for key in checkpoint.keys():
                    if "backbone" in key:
                        new_key = '.'.join(key.split('.')[1:])
                        new_dict[new_key] = checkpoint[key]
            else:
                new_dict = checkpoint
            model.load_state_dict(new_dict)
        else:
            model.load_state_dict(torch.load(cfg.shapley.model_path))


    with DupStdoutFileManager(str(Path(cfg.output_path) / (now_time + '.log'))) as _:
        print_easydict(cfg)
        if cfg.shapley.testdata:
            main(model, test_loader, nclass)
        else:
            main(model, train_loader, nclass)
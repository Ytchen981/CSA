import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
import time
from utils.parse_arg import cfg
from utils.utils import transform_train
import os
from pathlib import Path
from PIL import Image

class NFCBank(nn.Module):
    def __init__(self, conf_path = None, num_classes=10, conf_per_class=5000):
        """
        shape: feature shape (default: 128)
        K: bank size; number of confounder
        N: number of confounder for each sample
        """
        super(NFCBank, self).__init__()
        if conf_path is not None:
            cfg.train.conf_path = conf_path
        conf_set = [[] for _ in range(num_classes)]
        if isinstance(cfg.train.conf_path, list):
            for path in cfg.train.conf_path:
                for i in range(num_classes):
                    num = 0
                    shap_path = os.path.join(path, f"{i}")
                    assert Path(shap_path).exists()
                    files = os.listdir(shap_path)
                    for f in files:
                        if "neg.pt" in f:
                            num += 1
                    if num < conf_per_class:
                        conf_per_class = num
                for i in range(num_classes):
                    shap_path = os.path.join(path, f"{i}")
                    assert Path(shap_path).exists()
                    files = os.listdir(shap_path)
                    num = 0
                    for f in files:
                        if "neg.pt" in f and num < conf_per_class:
                            '''image = Image.open(os.path.join(shap_path, f)).convert('RGB')
                            image = transform_train(image)'''
                            image = torch.load(os.path.join(shap_path, f)).squeeze().unsqueeze(0)
                            conf_set[i].append(image)
                            num += 1
            for i in range(num_classes):
                conf_set[i] = torch.cat(conf_set[i], dim=0).unsqueeze(0)
            conf_set = torch.cat(conf_set, dim=0)
        else:
            path = cfg.train.conf_path
            for i in range(num_classes):
                num = 0
                shap_path = os.path.join(path, f"{i}")
                assert Path(shap_path).exists()
                files = os.listdir(shap_path)
                for f in files:
                    if "neg.pt" in f:
                        num += 1
                if num < conf_per_class:
                    conf_per_class = num
            for i in range(num_classes):
                shap_path = os.path.join(path, f"{i}")
                assert Path(shap_path).exists()
                files = os.listdir(shap_path)
                num = 0
                for f in files:
                    if "neg.pt" in f and num < conf_per_class:
                        '''image = Image.open(os.path.join(shap_path, f)).convert('RGB')
                        image = transform_train(image)'''
                        image = torch.load(os.path.join(shap_path, f)).squeeze().unsqueeze(0)
                        conf_set[i].append(image)
                        num += 1
                conf_set[i] = torch.cat(conf_set[i], dim=0).unsqueeze(0)
            conf_set = torch.cat(conf_set, dim=0)

        self.register_buffer("confounder_queue", torch.Tensor(conf_set))  #queue for confounder
        self.confounder_queue.squeeze()
        _, class_num, _, _, _ = self.confounder_queue.shape
        print(self.confounder_queue.shape)

        self.K = class_num
        self.N = cfg.train.class_conf_size
        self.nclass = num_classes

    @torch.no_grad()
    def batch_sample_set(self, x_s, label):
        bs_size = x_s.shape[0]
        conf_set = []
        index_list = [i for i in range(self.K)]
        for j in range(bs_size):
            conf_example = []
            for i in range(self.nclass):
                if i != label[j].item() and cfg.train.other_class_conf:
                    selected = np.random.choice(index_list, self.N, replace=False)
                    conf_class_choosed = self.confounder_queue[i][selected].clone()
                    conf_example.append(conf_class_choosed)
                elif i == label[j].item() and not cfg.train.other_class_conf:
                    selected = np.random.choice(index_list, self.N, replace=False)
                    conf_class_choosed = self.confounder_queue[i][selected].clone()
                    conf_example.append(conf_class_choosed)
            conf_example = torch.cat(conf_example, dim=0).unsqueeze(0)
            conf_set.append(conf_example)
        conf_set = torch.cat(conf_set, dim=0)

        return conf_set









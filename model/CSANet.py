import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import grad
# from util_func import make_backbone
from utils.parse_arg import cfg
# from utils.model_sl import load_model
from torch.nn import Linear, ReLU, CrossEntropyLoss
from model.NFCBank import NFCBank
import time

class CSANetwork(nn.Module):
    def __init__(self, backbone=None, num_classes=10, conf_per_class=5000, use_conf=False, mask_alpha=None):
        super(CSANetwork, self).__init__()
        self.backbone = backbone
        self.erb = NFCBank(num_classes=num_classes, conf_per_class=conf_per_class)
        self.test_CSA = use_conf
        self.mask_alpha = mask_alpha
        if self.test_CSA:
            print("Test with CSA")

    def forward(self, x):
        if self.test_CSA and not self.training:
            output = self.backbone(x)
            y_pred = output.max(1)[1].long()
            x, x_conf = self.split_x(x, y_pred)
            x_new = x + self.mask_alpha * x_conf
            preds = self.backbone(x_new)
        else:
            preds = self.backbone(x)
        return preds


    def split_x(self, x, y):
        # return a batch of confounder sets, each x_s has a confounder set with size of N, i.e., (B, N, ...)
        x_v_set = self.erb.batch_sample_set(x, y).cuda()
        # import pdb; pdb.set_trace()
        # approximate causal intervention using our causal attention network
        x_v_att = torch.mean(x_v_set, dim=1)
        # import pdb; pdb.set_trace()

        return x, x_v_att
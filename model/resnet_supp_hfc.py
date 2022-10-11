import torch
import torch.nn as nn
import torch.nn.functional as F
from core import HighFreqSuppress
from copy import deepcopy
from _jit_internal import weak_script_method
import cfg


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    @weak_script_method
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    @weak_script_method
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, c=64, num_classes=10, r=16):
        super(ResNet, self).__init__()
        self.in_planes = deepcopy(c)
        self.hfs_train = HighFreqSuppress(cfg.crop_size, cfg.crop_size, r)
        self.hfs_eval = HighFreqSuppress(cfg.img_size, cfg.img_size, r)
        self.head = head(c)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)

        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @weak_script_method
    def forward(self, x, training=True):
        if training:
            out = self.hfs_train(x)
        else:
            out = self.hfs_eval(x)

        out = self.head(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.mean(-1).mean(-1)
        out = self.linear(out)

        return out

def head(c):
    head = nn.Sequential(nn.Conv2d(3, c//2, kernel_size=5, stride=2, bias=False),
                         nn.BatchNorm2d(c//2),
                         nn.ReLU(True),
                         nn.Conv2d(c//2, c, kernel_size=3, bias=False),
                         nn.AvgPool2d(3, 2),
                         nn.BatchNorm2d(c),
                         nn.ReLU(True))
    return head

def ResNet18(dim, c=64, r=16):
    return ResNet(BasicBlock, [2,2,2,2], c, dim, r)

def ResNet34(dim, c=64, r=16):
    return ResNet(BasicBlock, [3,4,6,3], c, dim, r)

def ResNet50(dim, c=64, r=16):
    return ResNet(Bottleneck, [3,4,6,3], c, dim, r)


if __name__=='__main__':
    net = ResNet18(10, c=64)
    print(net)





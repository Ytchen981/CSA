import numpy as np
import time
import sys
import torch
import random
from torchvision import transforms
from utils.parse_arg import cfg

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_train_norm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

transform_test_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

ImageNet_train_transform = transforms.Compose ( [
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
] )
ImageNet_test_transform = transforms.Compose ( [
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor()
] )

TinyImageNet_train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)
TinyImageNet_test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor()
    ]
)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

import sys

class DupStdoutFileWriter(object):
    def __init__(self, stdout, path, mode):
        self.path = path
        self._content = ''
        self._stdout = stdout
        self._file = open(path, mode)

    def write(self, msg):
        while '\n' in msg:
            pos = msg.find('\n')
            self._content += msg[:pos + 1]
            self.flush()
            msg = msg[pos + 1:]
        self._content += msg
        if len(self._content) > 1000:
            self.flush()

    def flush(self):
        self._stdout.write(self._content)
        self._stdout.flush()
        self._file.write(self._content)
        self._file.flush()
        self._content = ''

    def __del__(self):
        self._file.close()

class DupStdoutFileManager(object):
    def __init__(self, path, mode='w+'):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self._stdout = sys.stdout
        self._file = DupStdoutFileWriter(self._stdout, self.path, self.mode)
        sys.stdout = self._file

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout

from easydict import EasyDict as edict

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(indent_cnt=0)
def print_easydict(inp_dict: edict):
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            print('{}{}:'.format(' ' * 2 * print_easydict.indent_cnt, key))
            print_easydict.indent_cnt += 1
            print_easydict(value)
            print_easydict.indent_cnt -= 1

        else:
            print('{}{}: {}'.format(' ' * 2 * print_easydict.indent_cnt, key, value))

@static_vars(indent_cnt=0)
def print_easydict_str(inp_dict: edict):
    ret_str = ''
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            ret_str += '{}{}:\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key)
            print_easydict_str.indent_cnt += 1
            ret_str += print_easydict_str(value)
            print_easydict_str.indent_cnt -= 1

        else:
            ret_str += '{}{}: {}\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key, value)

    return ret_str

class HighFreqSuppress(torch.nn.Module):
    def __init__(self, w, h, r):
        super(HighFreqSuppress, self).__init__()
        self.w = w
        self.h = h
        self.r = r
        self.templete()

    def templete(self):
        temp = np.zeros((self.w, self.h), "float32")
        cw = self.w // 2
        ch = self.h // 2
        if self.w % 2 == 0:
            dw = self.r
        else:
            dw = self.r + 1

        if self.h % 2 == 0:
            dh = self.r
        else:
            dh = self.r + 1

        temp[cw - self.r:cw + dw, ch - self.r:ch + dh] = 1.0
        temp = np.roll(temp, -cw, axis=0)
        temp = np.roll(temp, -ch, axis=1)
        temp = torch.tensor(temp)
        temp = temp.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        self.temp = temp
        if torch.cuda.is_available():
            self.temp = self.temp.cuda()

    def forward(self, x):
        x_hat = torch.rfft(x, 2, onesided=False)
        x_hat = x_hat * self.temp
        y = torch.irfft(x_hat, 2, onesided=False)

        return y

    def extra_repr(self):
        return 'feature_width={}, feature_height={}, radius={}'.format(self.w, self.h, self.r)
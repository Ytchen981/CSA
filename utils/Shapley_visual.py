import os
import collections
import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from matplotlib.colors import Normalize
import matplotlib as mlt
import math
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

def visual_shap(shap_value, w, h, img_path):
    max_value = torch.topk(shap_value, 1)[0].item()
    min_value = torch.topk(shap_value, 1, largest=False)[0].item()
    maximum = max([abs(max_value), abs(min_value)])

    shap_value = shap_value.view(w, h)
    plt.figure()
    norm = Normalize(vmin=-maximum, vmax=maximum)
    plt.imshow(shap_value, norm=norm, cmap=mlt.cm.bwr)
    plt.gca().get_yaxis().set_visible(False)  # 不显示y轴
    plt.gca().get_xaxis().set_visible(False)  # 不显示x轴
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(img_path, format='pdf')
    plt.close()

def visual_shap_w_tick(shap_value, w, h, img_path):
    max_value = torch.topk(shap_value, 1)[0].item()
    min_value = torch.topk(shap_value, 1, largest=False)[0].item()
    maximum = max([abs(max_value), abs(min_value)])

    x_spec = [i for i in range(w)]
    x_shift_spec = np.fft.fftshift(x_spec)
    y_spec = [i for i in range(h)]
    y_shift_spec = np.fft.fftshift(y_spec)


    shap_value = shap_value.view(w, h)
    plt.figure()
    norm = Normalize(vmin=-maximum, vmax=maximum)
    plt.imshow(shap_value, norm=norm, cmap=mlt.cm.bwr)
    #plt.gca().get_yaxis().set_visible(False)  # 不显示y轴
    #plt.gca().get_xaxis().set_visible(False)  # 不显示x轴
    plt.xticks(x_spec, x_shift_spec)
    plt.yticks(y_spec, y_shift_spec)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(img_path, format='svg')
    plt.close()

if __name__ == "__main__":
    x_spec = [i for i in range(16)]
    x_shift_spec = np.fft.fftshift(x_spec)
    y_spec = [i for i in range(16)]
    y_shift_spec = np.fft.fftshift(y_spec)
    print(x_spec)
    print(x_shift_spec)
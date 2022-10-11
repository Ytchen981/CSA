import torch
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
from utils.frequency import transform_fft, transform_ifft
import os
import time

# For an image of size C * H * W, sample a random order of H*W elements
# Generate (H * W + 1 ) masks, elements are masked one by one by the order
# Apply the mask on the image, (H * W + 1) * C * H * W
# The output of the model is of size (H * W + 1) * N
# Get the difference y[:-1] - y[1:]

def sample_mask(img_w, img_h, mask_w, mask_h, static_center=False):
    #return a sampled mask, a tensor of size ((mask_w*mask_h)+1, img_w, img_h)
    length = mask_w * mask_h + 1
    order = np.random.permutation(np.arange(0, mask_w * mask_h, 1))  # Sample an order
    mask = torch.ones(length, 3, mask_w, mask_h)
    mask = mask.view(length, 3, -1)
    for j in range(1, length):
        mask[j:, :, order[j - 1]] = 0
    mask = mask.view(length, 3, mask_w, mask_h)
    if static_center:
        mask[:, :, mask_w//2, mask_h//2] = 1
    mask = F.interpolate(mask.clone(), size=[img_w, img_h],mode="nearest").float()
    return mask, order

def getShapley_pixel(img, label, model, sample_times, mask_size, k=0):
    b, c, w, h = img.size()
    #assert b == 1 and label.size(0) == 1
    shap_value = torch.zeros((mask_size ** 2))
    with torch.no_grad():
        for i in range(sample_times):
            mask, order = sample_mask(w, h, mask_size, mask_size)
            base = img[k].expand(mask.size(0), 3, w, h).clone()
            masked_img = base * mask.cuda()
            output = model(masked_img)
            if torch.any(torch.isnan(output)):
                raise ValueError("NAN in output")
            y = output[:, label[k]]
            yy = y[:-1]
            dy = yy - y[1:]
            shap_value[order] += (dy.cpu())
        shap_value /= sample_times
    return shap_value

def getShapley_freq(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1, static_center=False, fix_masks=False, mask_path=None):
    b, c, w, h = img.size()
    length = mask_size ** 2 + 1
    # assert b == 1 and label.size(0) == 1
    shap_value = torch.zeros((mask_size ** 2))
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in range(sample_times // n_per_batch):
                maskes = []
                orders = []
                if not fix_masks:
                    for j in range(n_per_batch):
                        mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
                        maskes.append(mask)
                        orders.append(order)
                    maskes = torch.cat(maskes, 0)
                    assert maskes.size(0) == n_per_batch * length
                else:
                    maskes = torch.load(os.path.join(mask_path, f"mask_{i}.pth"))
                    orders = torch.load(os.path.join(mask_path, f"order_{i}.pth"))
                if split_n > 1:
                    base = transform_fft(img[k])
                    bs = maskes.size(0) // split_n
                    outputs = []
                    for j in range(maskes.size(0)//bs):
                        if j == maskes.size(0) // bs -1:
                            current_mask = maskes[j*bs:]
                        else:
                            current_mask = maskes[j*bs:(j+1) * bs]
                        masked_img = base.expand(current_mask.size(0), 3, w, h).clone() * current_mask
                        masked_img = transform_ifft(masked_img)
                        masked_img = masked_img.cuda()
                        masked_img = torch.clamp(masked_img, 0., 1.)
                        outputs.append(model(masked_img))
                    output = torch.cat(outputs, dim=0)
                else:
                    base = transform_fft(img[k]).expand(maskes.size(0), 3, w, h).clone()
                    masked_img = base * maskes
                    masked_img = transform_ifft(masked_img)
                    masked_img = masked_img.cuda()
                    masked_img = torch.clamp(masked_img, 0., 1.)
                    output = model(masked_img)
                for j in range(n_per_batch):
                    y = output[j * length:(j + 1) * length, label[k]]
                    yy = y[:-1]
                    dy = yy - y[1:]
                    if torch.any(torch.isnan(dy)):
                        raise ValueError("Nan in dy")
                    shap_value[orders[j]] += (dy.cpu())
                if i % 100 == 0:
                    print(f"{i}/{sample_times // n_per_batch}")
        shap_value /= sample_times//n_per_batch * n_per_batch
    return shap_value

def getShapley_freq_softmax(img, label, model, sample_times, mask_size, k=0, n_per_batch=1, split_n=1, static_center=False, fix_masks=False, mask_path=None):
    b, c, w, h = img.size()
    length = mask_size ** 2 + 1
    # assert b == 1 and label.size(0) == 1
    shap_value = torch.zeros((mask_size ** 2))
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in range(sample_times // n_per_batch):
                mask, order = sample_mask(w, h, mask_size, mask_size, static_center=static_center)
                base = transform_fft(img[k]).clone()
                base = base.expand(mask.size(0), c, w, h)
                masked_base = base * mask
                masked_img = transform_ifft(masked_base)
                masked_img = masked_img.cuda()
                masked_img = torch.clamp(masked_img, 0., 1.)
                output = F.softmax(model(masked_img), dim=1)
                y = output[:, label[k]]
                yy = y[:-1]
                dy = yy - y[1:]
                if torch.any(torch.isnan(dy)):
                    raise ValueError("Nan in dy")
                shap_value[order] += (dy.cpu())

                if i % 100 == 0:
                    print(f"{i}/{sample_times}")
        shap_value /= sample_times
    return shap_value

def gen_dis_list(mask_w, mask_h):
    dis_dict = dict()
    for i in range(mask_w * mask_h):
        dis = ((i // mask_w) - (mask_w / 2 - 0.5)) ** 2 + ((i % mask_w) - (mask_h / 2 - 0.5)) ** 2
        if f"{dis:.2f}" not in dis_dict.keys():
            dis_dict[f"{dis:.2f}"] = []
        dis_dict[f"{dis:.2f}"].append(i)
    dis = np.sort(np.array(list(dis_dict.keys()), dtype=float))
    return dis_dict, dis

def sample_mask_dict(img_w, img_h, mask_w, mask_h, dis_dict, keys):
    length = len(keys) + 1
    order = np.random.permutation(np.arange(0, len(keys), 1))  # Sample an order
    mask = torch.ones(length, 3, mask_w, mask_h)
    mask = mask.view(length, 3, -1)
    for j in range(1, length):
        points = dis_dict[f"{keys[order[j-1]]:.2f}"]
        mask[j:, :, points] = 0
    mask = mask.view(length, 3, mask_w, mask_h)
    mask = F.interpolate(mask.clone(), size=[img_w, img_h], mode="nearest").float()
    return mask, order

def getShapley_freq_dis(img, label, model, sample_times, mask_size, k=0):
    b, c, w, h = img.size()
    # assert b == 1 and label.size(0) == 1
    shap_value = torch.zeros((mask_size ** 2))
    dis_dict, dis = gen_dis_list(mask_size, mask_size)
    with torch.no_grad():
        for i in range(sample_times):
            mask, order = sample_mask_dict(w, h, mask_size, mask_size, dis_dict, dis)
            base = transform_fft(img[k]).expand(mask.size(0), 3, w, h).clone()
            masked_img = base * mask
            masked_img = transform_ifft(masked_img)
            masked_img = masked_img.cuda()
            output = model(masked_img)
            if torch.any(torch.isnan(output)):
                masked_img = torch.clamp(masked_img, 0., 1.)
                output = model(masked_img)
                assert not torch.any(torch.isnan(output))
            y = output[:, label[k]]
            yy = y[:-1]
            dy = yy - y[1:]
            if torch.any(torch.isnan(dy)):
                raise ValueError("Nan in dy")
            for i in range(len(dy)):
                shap_value[dis_dict[f"{dis[order[i]]:.2f}"]] += (dy[i].cpu())
        shap_value /= sample_times
    return shap_value

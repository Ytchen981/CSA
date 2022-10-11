import torch
import numpy as np
import torch.nn.functional as F
from utils.frequency import transform_fft, transform_ifft

def sample_mask_interaction(img_w, img_h, mask_w, mask_h, center):
    #return a sampled mask, a tensor of size ((mask_w*mask_h)+1, img_w, img_h)
    length = mask_w * mask_h
    order = np.arange(0, mask_w * mask_h, 1)
    order = np.delete(order, center)
    order = np.random.permutation(order)  # Sample an order
    mask = torch.ones(length, 3, mask_w, mask_h)
    mask = mask.view(length, 3, -1)
    for j in range(1, length):
        mask[j:, :, order[j - 1]] = 0
    wo_mask = mask.clone()
    mask = mask.view(length, 3, mask_w, mask_h)
    mask = F.interpolate(mask.clone(), size=[img_w, img_h],mode="nearest").float()

    wo_mask[:, :, center] = 0
    wo_mask = wo_mask.view(length, 3, mask_w, mask_h)
    wo_mask = F.interpolate(wo_mask, size=[img_w, img_h],mode="nearest").float()
    return mask, wo_mask, order

def getInteraction_freq(img, label, model, sample_times, mask_size, center, k=0):
    b, c, w, h = img.size()
    # assert b == 1 and label.size(0) == 1
    interaction = torch.zeros((mask_size ** 2))
    with torch.no_grad():
        for i in range(sample_times):
            mask, wo_mask, order = sample_mask_interaction(w, h, mask_size, mask_size, center)
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

            wo_masked_img = base * wo_mask
            wo_masked_img = transform_ifft(wo_masked_img)
            wo_masked_img = wo_masked_img.cuda()
            wo_output = model(wo_masked_img)
            if torch.any(torch.isnan(output)):
                wo_masked_img = torch.clamp(wo_masked_img, 0., 1.)
                wo_output = model(wo_masked_img)
                assert not torch.any(torch.isnan(wo_output))
            wo_y = wo_output[:, label[k]]
            wo_yy = wo_y[:-1]
            wo_dy = wo_yy - wo_y[1:]

            if torch.any(torch.isnan(dy)):
                raise ValueError("Nan in dy")
            if torch.any(torch.isnan(wo_dy)):
                raise ValueError("Nan in wo_dy")
            interaction[order] += ((dy - wo_dy).cpu())
        interaction /= sample_times
    return interaction
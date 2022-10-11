import numpy as np
import torch

def transform_fft(img):
    if img.dim() > 3:
        img = img.view(-1, img.size(-3), img.size(-2), img.size(-1))
        result = []
        for i in range(img.size(0)):
            tmp = np.array(img[i].cpu())
            tmp = np.fft.fft2(tmp)
            tmp = np.fft.fftshift(tmp)
            result.append(torch.tensor(tmp).unsqueeze(0))
    else:
        img = np.array(img.cpu())
        img = np.fft.fft2(img)
        img = np.fft.fftshift(img)
        return torch.tensor(img)

    result = torch.cat(result, dim=0)
    return result

def transform_ifft(img):
    if img.dim() > 3:
        img = img.view(-1, img.size(-3), img.size(-2), img.size(-1))
        result = []
        for i in range(img.size(0)):
            tmp = np.array(img[i].cpu())
            tmp = np.fft.ifft2(np.fft.ifftshift(tmp))
            if np.abs(np.sum(np.imag(tmp))) > 1e-5:
                raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")
            result.append(torch.tensor(np.real(tmp)).float().unsqueeze(0))
    else:
        img = np.array(img.cpu())
        img = np.fft.ifft2(np.fft.ifftshift(img))
        if np.abs(np.sum(np.imag(img))) > 1e-5:
            raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")
        return torch.tensor(np.real(img)).float()

    result = torch.cat(result, dim=0)
    return result



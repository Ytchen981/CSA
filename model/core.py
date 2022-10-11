import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _jit_internal import weak_script_method


def squared_l2_norm(x):
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).mean(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

class Trades:
    def __init__(self, step_size=0.003, epsilon=0.047, perturb_steps=5, beta=1.0):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")

    def reset_steps(self, k):
        self.perturb_steps = k

    @weak_script_method
    def PGD_L2(self, model, x_natural, logits):
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()
        prob = F.softmax(logits, dim=-1)

        for _ in range(self.perturb_steps):
            with torch.enable_grad():
                x_adv.requires_grad_()
                loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0].detach()
            grad /= l2_norm(grad).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8
            x_adv = x_adv.detach() + self.step_size * grad

            delta = x_adv - x_natural
            delta_norm = l2_norm(delta)
            cond = delta_norm > self.epsilon
            delta[cond] *= self.epsilon / delta_norm[cond].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_adv = x_natural + delta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def PGD_Linf(self, model, x_natural, logits):
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()
        prob = F.softmax(logits, dim=-1)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0].detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def loss(self, model, logits, x_adv, labels, optimizer):
        model.train()
        optimizer.zero_grad()
        prob = F.softmax(logits, dim=-1)
        loss_natural = F.cross_entropy(logits, labels)
        loss_robust = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
        loss = loss_natural + self.beta * loss_robust

        return loss


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

    @weak_script_method
    def forward(self, x):
        x_hat = torch.rfft(x, 2, onesided=False)
        x_hat = x_hat * self.temp
        y = torch.irfft(x_hat, 2, onesided=False)

        return y

    def extra_repr(self):
        return 'feature_width={}, feature_height={}, radius={}'.format(self.w, self.h, self.r)
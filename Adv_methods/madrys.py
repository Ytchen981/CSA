import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def madrys_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    adv_acc = (logits.max(1)[1] == y).sum().item() / y.size(0)

    return loss, adv_acc

def pgd(model,
              x_natural,
              y,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf',
            is_normalize=False):
    if is_normalize:
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
        upper_limit = ((1 - mu) / std)
        lower_limit = ((0 - mu) / std)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            if is_normalize:
                x_adv = torch.min(torch.max(x_adv, lower_limit), upper_limit)
            else:
                x_adv = torch.clamp(x_adv, 0., 1.)
    else:
        if is_normalize:
            x_adv = torch.min(torch.max(x_adv, lower_limit), upper_limit)
        else:
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv
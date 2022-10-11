import os
import torch
# import numpy as np
# import random
import xlwt
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from pathlib import Path
import apex.amp as amp
from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast as autocast, GradScaler

from utils.parse_arg import parse_args, cfg
from utils.utils import set_seed, DupStdoutFileManager, print_easydict, transform_train, transform_test, TinyImageNet_train_transform, TinyImageNet_test_transform
from Adv_methods.madrys import madrys_loss, pgd
from Adv_methods.trades import trades_loss
from utils.frequency import transform_ifft, transform_fft
import math
from model.CSANet import CSANetwork
from model.wideresnet import WideResNet

parse_args('Training model.')

#from model.resnet import *
from model.vgg import *

def conf_alpha_schedule(epoch, type='linear', steep=False, maximum=None):
    if maximum is None:
        maximum = cfg.scheduler.milestones[0]+2
    if epoch > maximum:
        if steep:
            return 10
        else:
            return 1
    if type == 'linear':
        return epoch / maximum
    elif type == 'Cosine':
        return 1 - math.cos(math.pi * epoch / (2 * maximum))
    elif type == 'constant':
        return 1
    else:
        raise KeyError(f"Unknown type {type}")

def train_epoch(model, dataloader, optimizer, epoch, tfboard_writer, scaler):
    model.train()
    model.cuda()
    Acc = 0.
    Adv_Acc = 0.
    total_loss = 0.
    loss_dict = dict()
    acc_dict = dict()
    for i,(X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()
        X, X_conf = model.split_x(X, y)
        optimizer.zero_grad()
        logit = model(X)
        #loss = F.cross_entropy(logit, y)
        with autocast():
            if cfg.dataset == 'imagenet':
                X_conf = torch.clamp(X + cfg.train.mask_alpha * conf_alpha_schedule(i + (epoch-1)*len(dataloader), cfg.train.conf_schedule_type, cfg.train.conf_steep, maximum=cfg.scheduler.epochs * len(dataloader)) * X_conf, 0., 1.) - X
            else:
                X_conf = torch.clamp(X + cfg.train.mask_alpha * conf_alpha_schedule(epoch, cfg.train.conf_schedule_type, cfg.train.conf_steep) * X_conf, 0., 1.) - X
            if cfg.adv.loss_type == "madrys":
                conf_loss, conf_acc = madrys_loss(model=model,
                                                  x_natural=X_conf + X,
                                                  y=y,
                                                  optimizer=optimizer,
                                                  step_size=cfg.adv.train_step_size,
                                                  epsilon=cfg.adv.train_epsilon,
                                                  perturb_steps=cfg.adv.train_num_steps,
                                                  distance='l_inf')
            elif cfg.adv.loss_type == "trades":
                conf_loss, conf_acc = trades_loss(model=model,
                                                  x_natural=X,
                                                  y=y,
                                                  optimizer=optimizer,
                                                  step_size=cfg.adv.train_step_size,
                                                  epsilon=cfg.adv.train_epsilon,
                                                  perturb_steps=cfg.adv.train_num_steps,
                                                  beta=cfg.adv.train_beta,
                                                  distance='l_inf',
                                                  conf=X_conf)
            loss = conf_loss
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc = (logit.max(1)[1] == y).sum().item() / y.size(0)

        Acc += acc
        Adv_Acc += conf_acc
        total_loss += loss.item()

        loss_dict['loss'] = loss.item()
        tfboard_writer.add_scalars('training loss', loss_dict, epoch * cfg.train.batchsize + i)

        acc_dict['accuracy'] = acc
        tfboard_writer.add_scalars(
            'training accuracy',
            acc_dict,
            epoch * cfg.train.batchsize + i
        )
        if cfg.dataset == 'imagenet':
            if i % 100 == 0:
                print('Train Epoch: {} batch {} \ {} \t CE: {:.4f} Acc: {:.2f}% Adv Acc: {:.2f}'.format(epoch, i, len(dataloader),
                                                                                                total_loss / (i + 1),
                                                                                                100 * Acc / (i + 1),
                                                                                                100 * Adv_Acc / (i + 1)))
    print('Train Epoch: {} \t CE: {:.4f} Acc: {:.2f}% Adv Acc: {:.2f}'.format(epoch,  total_loss / (i + 1),
                                                              100 * Acc / (i + 1), 100 * Adv_Acc / (i + 1)))
    return total_loss / len(dataloader), 100 * Adv_Acc / len(dataloader)

def test_epoch(model, dataloader, epoch, tfboard_writer):
    model.eval()
    model.cuda()
    Acc = 0.
    Adv_Acc = 0.
    total_loss = 0.
    loss_dict = dict()
    acc_dict = dict()

    for i, (X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()
        Adv_X =  pgd(model=model,
                           x_natural=X,
                           y=y,
                           step_size=cfg.adv.test_step_size,
                           epsilon=cfg.adv.test_epsilon,
                           perturb_steps=cfg.adv.test_num_steps)
        with torch.no_grad():
            logit = model(X)
            loss = F.cross_entropy(logit, y)
            acc = (logit.max(1)[1] == y).sum().item() / y.size(0)

            adv_logit = model(Adv_X)
            adv_loss = F.cross_entropy(adv_logit, y)
            adv_acc = (adv_logit.max(1)[1] == y).sum().item() / y.size(0)

            Acc += acc
            Adv_Acc += adv_acc
            total_loss += loss.item()

            loss_dict['testing loss'] = loss.item()
            tfboard_writer.add_scalars('testing loss', loss_dict, epoch * cfg.train.batchsize + i)

            acc_dict['testing accuracy'] = acc
            tfboard_writer.add_scalars(
                'testing accuracy',
                acc_dict,
                epoch * cfg.train.batchsize + i
            )
        if cfg.dataset == 'imagenet':
            if i % 100 == 0:
                print('Test Epoch: {} batch {} \ {} \t CE: {:.4f} Acc: {:.2f}% Adv Acc: {:.2f}'.format(epoch, i+1, len(dataloader),
                                                                                         total_loss / (i + 1),
                                                                                         100 * Acc / (i + 1),
                                                                                         100 * Adv_Acc / (i + 1)))
    print('Test Epoch: {} \t CE: {:.4f} Acc: {:.2f}% Adv Acc: {:.2f}'.format(epoch,
                                                                                       total_loss / (i + 1),
                                                                                       100 * Acc / (i + 1),
                                                                                 100 * Adv_Acc / (i + 1)))
    return total_loss / len(dataloader), 100 * Adv_Acc / len(dataloader)

def main(model, trainloader, testloader, optimizer, scheduler, tfboard_writer, scaler):
    train_l, train_a, test_l, test_a = [], [], [], []
    best_acc = 0
    #test_loss, test_acc = test_epoch(model, testloader, 0, tfboard_writer)
    for epoch in range(cfg.scheduler.start_epoch+1, cfg.scheduler.epochs+1):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, epoch, tfboard_writer, scaler)
        test_loss, test_acc = test_epoch(model, testloader, epoch, tfboard_writer)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(cfg.output_path, "params")
            if not Path(model_path).exists():
                Path(model_path).mkdir(parents=True)
            model_path = os.path.join(model_path, f"model_best_acc.pt")
            torch.save(model.state_dict(), model_path)
        if (epoch) % 10 == 0 or (epoch) == cfg.scheduler.epochs or cfg.dataset == 'imagenet':
            model_path = os.path.join(cfg.output_path, "params")
            if not Path(model_path).exists():
                Path(model_path).mkdir(parents=True)
            opt_path = os.path.join(model_path, f"opt_epoch{epoch}.tar")
            model_path = os.path.join(model_path, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
        train_l.append(train_loss)
        train_a.append(train_acc)
        test_l.append(test_loss)
        test_a.append(test_acc)
        plot_loss(train_l, train_a, test_l, test_a)

def plot_loss(train_loss, train_acc, test_loss, test_acc):
    x = [i for i in range(len(train_loss))]
    plt.figure()
    plt.subplot(211)
    plt.plot(x, train_loss, color='red', label='train')
    plt.plot(x, test_loss, color='blue', label='test')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(212)
    plt.plot(x, train_acc, color='red', label='train')
    plt.plot(x, test_acc, color='blue', label='test')
    plt.xlabel("epoch")
    plt.ylabel("error_adv")
    plt.legend()

    plt.savefig(os.path.join(cfg.output_path, "loss_acc.png"))
    plt.close()


if __name__ == '__main__':
    set_seed(cfg.train.seed)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.output_path) / 'tensorboard' / 'training_{}'.format(now_time)))

    if cfg.dataset == "CIFAR10" or cfg.dataset == "cifar10":
        nclass=10
        if cfg.train.norm:
            trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train_norm)
            testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test_norm)
        else:
            trainset = datasets.CIFAR10(root=cfg.data_path, train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=cfg.data_path, train=False, download=True, transform=transform_test)
    elif cfg.dataset == "cifar100":
        nclass = 100
        trainset = datasets.CIFAR100(root=cfg.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=cfg.data_path, train=False, download=True, transform=transform_test)
    else:
        raise KeyError(f"Unknown model {cfg.dataset}")
    train_loader = DataLoader(trainset, batch_size=cfg.train.batchsize, shuffle=True, pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.num_workers)
    test_loader = DataLoader(testset, batch_size=cfg.train.batchsize, shuffle=False, pin_memory=cfg.dataloader.pin_memory, num_workers=cfg.dataloader.num_workers)


    if cfg.model == "ResNet18":
        from model.resnet import *
        backbone = ResNet18(num_classes=nclass)
    elif "WRN" in cfg.model:
        model_name_list = cfg.model.split('-')
        depth = int(model_name_list[1])
        width = int(model_name_list[2])
        backbone = WideResNet(depth=depth, num_classes=nclass, widen_factor=width)
    elif cfg.model == "vgg16_bn":
        backbone = vgg16_bn(num_classes=nclass)
    else:
        raise KeyError(f"Unknown model {cfg.model}")


    model = CSANetwork(backbone=backbone, num_classes=nclass, conf_per_class=cfg.train.conf_per_class).cuda()

    optimizer = optim.SGD(params=model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum,
                           weight_decay=cfg.train.weight_decay)

    if cfg.scheduler.start_epoch>0:
        model_state_dict = torch.load(os.path.join(cfg.resume_path, "params", f"model_epoch{cfg.scheduler.start_epoch}.pt"))
        opt_state_dict = torch.load(os.path.join(cfg.resume_path, "params", f"opt_epoch{cfg.scheduler.start_epoch}.tar"))
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(opt_state_dict)

    '''amp_args = dict(opt_level=cfg.train.opt_level, loss_scale=cfg.train.loss_scale, verbosity=False)
    if cfg.train.opt_level == 'O2':
        amp_args['master_weights'] = cfg.master_weights

    model, optimizer = amp.initialize(model, optimizer, **amp_args)'''
    scaler = GradScaler()

    if cfg.scheduler.start_epoch>0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones,
                                               gamma=cfg.scheduler.lr_decay, last_epoch=cfg.scheduler.start_epoch)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones,
                                                   gamma=cfg.scheduler.lr_decay)

    with DupStdoutFileManager(str(Path(cfg.output_path) / (now_time + '.log'))) as _:
        print_easydict(cfg)
        main(model, train_loader, test_loader, optimizer, scheduler, tfboardwriter, scaler)

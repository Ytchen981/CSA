import os
import argparse
import datetime
from easydict import EasyDict as edict
from pathlib import Path
# import numpy as np
import importlib
# import src.dataset

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.model = 'ResNet18'
__C.dataset = 'CIFAR10'
__C.output_path = ''
__C.data_path = '~/data/'
__C.resume_path = ''
__C.resume_path_list = []
__C.Suppressing_High_Freq = False
__C.Suppressing_High_Freq_pth = ''
__C.HFCr = 16

__C.train = edict()
__C.train.norm = False
__C.train.seed = 0
__C.train.batchsize = 128
__C.train.opt_level = 'O1'
__C.train.loss_scale = '1.0'
__C.train.lr = 0.1
__C.train.momentum = 0.9
__C.train.weight_decay = 5e-4
__C.train.log_period = 100
__C.train.mix_ratio = 1.

__C.train.mask_freq = False
__C.train.mask_alpha = 0.
__C.train.freq_min = 0.
__C.train.freq_max = 1.
__C.train.conf_schedule_type = 'linear'
__C.train.conf_steep = False

__C.train.conf_path = ''
__C.train.conf_per_class = 5000
__C.train.class_conf_size = 10
__C.train.conf_prior_dist = 'uniform'
__C.train.conf_no_att = True
__C.train.conf_loss_alpha = 0.2
__C.train.other_class_conf = False

__C.test = edict()

__C.test.use_conf = False
__C.test.mask_alpha = 0.

__C.dataloader = edict()
__C.dataloader.num_workers = 4
__C.dataloader.pin_memory = True

__C.scheduler = edict()
__C.scheduler.start_epoch = 0
__C.scheduler.epochs = 160
__C.scheduler.type = "multistep"
__C.scheduler.milestones = [80, 120]
__C.scheduler.lr_decay = 0.1

__C.shapley = edict()
__C.shapley.sample_times = 1000
__C.shapley.start_num = 0
__C.shapley.start_class = 0
__C.shapley.end_class = 999
__C.shapley.num_per_class = 50
__C.shapley.mask_size = 16
__C.shapley.model_path = ''
__C.shapley.shap_avg_path = ''
__C.shapley.testdata = True
__C.shapley.n_per_batch = 1
__C.shapley.split_n = 1
__C.shapley.adv_sample = False

__C.shapley.fix_mask = False
__C.shapley.static_center = True
__C.shapley.get_freq_by_dis = False
__C.shapley.get_false = False

__C.adv = edict()
__C.adv.loss_type = 'madrys'
__C.adv.train_beta = 6.
__C.adv.train_step_size = 2.
__C.adv.train_epsilon = 8.
__C.adv.train_num_steps = 10
__C.adv.test_step_size = 2.
__C.adv.test_epsilon = 8.
__C.adv.test_num_steps = 20

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            if type(b[k]) is float and type(v) is int:
                v = float(v)
            else:
                if not k in ['CLASS']:
                    raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.full_load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', '--config', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--prefix', default=None, type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.prefix is None:
        exp_name = ""
    else:
        exp_name = f"{args.prefix}_"
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)
            f = f.split('/')[-1]
            exp_name = exp_name + f"{f}"

    now_day = datetime.datetime.now().strftime('%Y-%m-%d')
    outp_path = os.path.join('output', now_day, exp_name)
    cfg_from_list(['output_path', outp_path])
    if not Path(cfg.output_path).exists():
        Path(cfg.output_path).mkdir(parents=True)
    cfg.adv.train_step_size /= 255
    cfg.adv.train_epsilon /= 255
    cfg.adv.test_step_size /= 255
    cfg.adv.test_epsilon /= 255





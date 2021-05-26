import os
import copy
import shutil
from pathlib import Path
from easydict import EasyDict as edict

import yaml
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

def make_dir(*paths, rmdir=False):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
        if os.path.isdir(path) and rmdir:
            shutil.rmtree(path)
            os.makedirs(path)

def create_out_folder(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    dirs = os.listdir(output_path)
    dirs = [x for x in dirs if os.path.isdir(os.path.join(output_path, x)) and 'exp' in x]

    if not dirs:
        output_path = Path(output_path, 'exp_01')
    else:
        new_num = max([int(x.split('_')[1]) for x in dirs]) + 1
        output_path = Path(output_path, f'exp_{new_num:02d}')
    os.makedirs(output_path)
    return output_path

def copy_files(out_dir, files):
    out_dir = Path(out_dir, 'source_files')
    make_dir(out_dir)
    for file in files:
        shutil.copyfile(file, Path(out_dir, file))


def load_yaml(filename):
    with open(filename, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = list_to_numpy(cfg)
    return edict(cfg)

def save_state_dict(latent, optimizer, iter, losses, out_path):
    save_dict = {'latent': latent.detach(),
                 'optimizer': optimizer.state_dict(),
                 'iter': iter,
                 'losses': losses}
    torch.save(save_dict, out_path)

def load_state_dict(in_path):
    load_dict = torch.load(in_path)
    latent = load_dict['latent']
    optimizer_state_dict = load_dict['optimizer']
    iter = load_dict['iter']
    losses = load_dict['losses']

    return latent, optimizer_state_dict, iter, losses

def debug_cfg(cfg):
    cfg['N_STEPS'] = 60
    cfg['CLASSIFICATION_START'] = 20
    cfg['SEG_START'] = 40
    cfg['NOISE_START'] = 50

def save_img(img, *args, **kwargs):
    save_image(img.clone().detach().cpu(), *args, **kwargs)

def set_cfg(cfg, attr):
    if not 'hair' in attr:  # only optimize shape term for hair
        cfg.LOSS.weights.seg = 0
    if attr not in ('black_hair', 'blond_hair', 'brown_hair', 'gray_hair'):
        cfg.DYNAMIC_MASKING = False

    cfg.OUTPUT_PATH = Path(cfg.OUTPUT_PATH, attr)
    return cfg

def interpolate(img, size):
    if type(size) == tuple:
        assert size[0] == size[1]
        size = size[0]

    orig_size = img.size(3)
    if size < orig_size:
        mode = 'area'
    else:
        mode = 'bilinear'
    return F.interpolate(img, (size, size), mode=mode)

def list_to_numpy(cfg):
    for k, v in cfg.items():
        if type(v) is list:
            cfg[k] = np.array(sorted(v))
    return cfg

def overwrite_cfg(cfg, args):
    for arg_k, arg_v in vars(args).items():
        arg_k = arg_k.upper()
        if arg_k in cfg and arg_v is not None:
            cfg[arg_k] = arg_v
    return cfg

def load_config(filename, args):
    cfg = load_yaml(filename)
    cfg = update_cfg_based_on_attributes(cfg, args.attribute)
    cfg = overwrite_cfg(cfg, args)

    cfg.DEVICE = 'cuda:0' if cfg.GPU else 'cpu'
    return cfg

def save_config(cfg):
    for k, v in cfg.items():
        if isinstance(v, np.ndarray):
            cfg[k] = v.tolist()
        elif isinstance(v, edict):
            cfg[k] = dict(v)

    with open(os.path.join(cfg.OUTPUT_PATH, 'cfg_updated.yaml'), 'w') as f:
        yaml.dump(cfg, f)

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != 1024:
        img = interpolate(img, 1024)
    return img

def update_cfg_based_on_attributes(cfg, attributes):
    if not isinstance(attributes, list):
        attributes = [attributes]

    is_hair_shape = is_hair_color = is_hair = False
    for attribute in attributes:
        assert attribute in cfg.ALL_ATTRIBUTES, 'This attribute is not available, refer to config.yml for ' \
                                                'available attributes'

        is_hair_shape |= attribute in ('wavy_hair', 'straight_hair')
        is_hair_color |= attribute in ('black_hair', 'brown_hair', 'blond_hair', 'gray_hair')
        is_hair |= is_hair_color or is_hair_shape

    cfg = copy.deepcopy(cfg)
    if not is_hair:
        cfg.LOSS.weights.seg = 0
    if is_hair_shape:
        cfg.MODELS.SEGMENTATOR.UPDATE_SHAPE = True

    cfg.ATTRIBUTES = attributes
    return cfg

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
        cfg.loss.weights.seg = 0
    if attr not in ('black_hair', 'blond_hair', 'brown_hair', 'gray_hair'):
        cfg.dynamic_masking = False

    cfg.output_path = Path(cfg.output_path, attr)
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

def save_config(cfg):
    for k, v in cfg.items():
        if isinstance(v, np.ndarray):
            cfg[k] = v.tolist()
        elif isinstance(v, edict):
            cfg[k] = dict(v)

    with open(os.path.join(cfg.output_path, 'cfg_updated.yaml'), 'w') as f:
        yaml.dump(cfg, f)

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = TF.to_tensor(img)
    img = img.unsqueeze(0)
    if img.size(-1) != 1024:
        img = interpolate(img, 1024)
    return img


class Config:
    def __init__(self, filename, attribute, size: float=0, e4e_init: bool=False, force_global: bool=False,
                       blend_option='include_skin'):
        self.attribute = attribute
        self.size = size
        self.e4e_init = e4e_init
        self.force_global = force_global
        self.blend_option = blend_option

        self.cfg = load_yaml(filename)
        self.cfg.size = size
        self.cfg.blend_option = blend_option

        self.cfg.attributes = [self.attribute]
        self.set_hair_config()
        self.set_attr_config()

    def set_global_cfg(self):
        self.cfg.models.is_local = False
        self.cfg.e4e_init = True
        self.cfg.loss.weights.recon = self.cfg.loss.weights.lpips
        self.cfg.dynamic_masking = False
        self.cfg.loss.start_steps.classf = 0
        self.cfg.optimizer.reinit = False
    
    def set_local_cfg(self):
        self.cfg.models.is_local = True
        self.cfg.e4e_init = self.e4e_init
        self.cfg.loss.weights.recon = self.cfg.loss.weights.mse

    def set_hair_config(self):
        is_hair_shape = is_hair_color = is_hair = False
        is_hair_shape |= self.attribute in ('wavy_hair', 'straight_hair')
        is_hair_color |= self.attribute in ('black_hair', 'brown_hair', 'blond_hair', 'gray_hair')
        is_hair |= is_hair_color or is_hair_shape

        if not is_hair:
            self.cfg.loss.weights.seg = 0
        if is_hair_shape:
            self.cfg.models.update_shape = True

        return is_hair, is_hair_shape

    def set_attr_config(self):
        if self.force_global:
            is_local = False
        else:
            is_local = self.attribute in self.cfg.local_attributes

        if is_local:
            self.set_local_cfg()
        else:
            self.set_global_cfg()
        
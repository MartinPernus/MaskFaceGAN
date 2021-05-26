from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.segmentation import deeplabv3_resnet50

class FaceParser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mask_modes = ['mse', 'shape', 'blend', 'dynamic']
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=cfg.N_CLASSES)
        self.model = load_lightning_dict(self.model, cfg.CKPT)
        self.all_masks = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']
        self.mask2idx = {k: self.all_masks.index(k) for k in self.all_masks}

        self.downsample_cfg = {'size': (512, 512), 'mode': 'area'}
        self.upsample_cfg = {'size': (1024, 1024), 'mode': 'bilinear'}
        self.mask_idxs = {'mse': [self.mask2idx['skin']],
                          'blend': [self.mask2idx['skin']],
                          'shape': [],
                          'dynamic': []}

    def set_idx_list(self, attributes):
        for attr in attributes:
            self.set_idx(attr)

    def set_idx(self, attribute):  # tole je koristno samo za potem, ko optimiziras shape

        if attribute in ['wearing_lipstick', 'mouth_slightly_open', 'smiling', 'big_lips']:
            target_name = 'mouth'
        elif attribute in ['bushy_eyebrows', 'arched_eyebrows']:
            target_name = 'eyebrows'
        elif attribute in ['narrow_eyes']:
            target_name = 'eyes'
        elif attribute in ['pointy_nose', 'big_nose']:
            target_name = 'nose'
        elif attribute in ['black_hair', 'brown_hair', 'blond_hair', 'gray_hair', 'wavy_hair', 'straight_hair']:
            target_name = 'hair'
        else:
            raise ValueError('attribute not found')

        target_idx = self.mask2idx[target_name]
        self.mask_idxs['blend'] += [target_idx]
        self.mask_idxs['shape'] += [target_idx]
        self.mask_idxs['dynamic'] += [target_idx]

        if attribute in ('straight_hair', 'wavy_hair'):  # hair shape requires careful treatment, especially with ear component
            self.mask_idxs['mse'] += [self.mask2idx['ears']]
            self.mask_idxs['blend'] += [self.mask2idx['ears']]
            self.mask_idxs['shape'] += [self.mask2idx['ears']]

    def forward(self, img, mode: str):
        if img.size(-1) != 512:
            img = F.interpolate(img, **self.downsample_cfg)
        img = img * 2 - 1
        parsed_face = F.softmax(self.model(img)['out'], dim=1)
        parsed_face = parsed_face[:, self.mask_idxs[mode]]

        if mode != 'dynamic':
            parsed_face = torch.sum(parsed_face, dim=1, keepdim=True)

        if mode != 'shape':
            parsed_face = F.interpolate(parsed_face, **self.upsample_cfg)
        return parsed_face


def load_lightning_dict(model, ckpt_file):
    thisdir = Path(__file__).parent
    state_dict = torch.load(thisdir / ckpt_file, map_location=torch.device('cuda:0'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'model.' in k:
            new_state_dict[k[6:]] = v
        elif 'loss.weight' in k:
            continue
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.eval().requires_grad_(False)
    return model

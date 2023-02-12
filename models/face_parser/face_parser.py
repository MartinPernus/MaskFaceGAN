import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


def attr2regions(attr):
    if attr in ['wearing_lipstick', 'mouth_slightly_open', 'smiling', 'big_lips']:
        regions = ['mouth']
    elif attr in ['bushy_eyebrows', 'arched_eyebrows']:
        regions = ['eyebrows']
    elif attr in ['narrow_eyes']:
        regions = ['eyes']
    elif attr in ['pointy_nose', 'big_nose']:
        regions = ['nose']
    elif attr in ['black_hair', 'brown_hair', 'blond_hair', 'gray_hair', 'wavy_hair', 'straight_hair']:
        regions = ['hair']
    else:
        regions = ['mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'ears']
    return regions

class FaceParser(nn.Module):
    def __init__(self, blend_option):
        super().__init__()
        self.mask_modes = ['recon', 'shape', 'blend', 'dynamic']
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=9)

        self.all_masks = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']
        self.mask2idx = {k: self.all_masks.index(k) for k in self.all_masks}

        self.downsample_cfg = {'size': (512, 512), 'mode': 'area'}
        self.upsample_cfg = {'size': (1024, 1024), 'mode': 'bilinear'}

        self.mask_idxs = {'recon': [self.mask2idx['skin']],
                          'blend': [self.mask2idx['skin']] if blend_option == 'include_skin' else [],
                          'shape': [],
                          'dynamic': [],
                          'global': [self.mask2idx[attr] for attr in self.all_masks[1:]]
                          }
        self.load()
        self.requires_grad_(False)
        self.eval()

    def load(self):
        thisdir = os.path.dirname(__file__)
        ckpt = torch.load(os.path.join(thisdir, 'state.pt'), map_location='cpu')
        del ckpt['loss.weight']
        self.load_state_dict(ckpt)

    def set_idx_list(self, attributes):
        for attr in attributes:
            self.set_idx(attr)

        for mask, idxs in self.mask_idxs.items():
            self.mask_idxs[mask] = list(set(idxs))

    def set_idx(self, attribute):   
        target_regions = attr2regions(attribute)
        target_idxs = [self.mask2idx[tr] for tr in target_regions]

        self.mask_idxs['blend'] += target_idxs
        self.mask_idxs['shape'] += target_idxs
        self.mask_idxs['dynamic'] += target_idxs

        if attribute in ('straight_hair', 'wavy_hair'):  # hair shape requires careful treatment, especially with ear component
            self.mask_idxs['recon'] += [self.mask2idx['ears']]
            self.mask_idxs['blend'] += [self.mask2idx['ears']]
            self.mask_idxs['shape'] += [self.mask2idx['ears']]

    def forward(self, img, mode: str):
        if img.size(-1) != 512:
            img = F.interpolate(img, **self.downsample_cfg)
        img = img * 2 - 1
        parsed_face = F.softmax(self.model(img)['out'], dim=1)
        parsed_face = parsed_face[:, self.mask_idxs[mode]]

        parsed_face = torch.sum(parsed_face, dim=1, keepdim=True)

        if mode != 'shape':
            parsed_face = F.interpolate(parsed_face, **self.upsample_cfg)
        return parsed_face

    @torch.no_grad()
    def get_image_masks(self, images, is_local):
        assert images.size(2) == images.size(3)
        masks = {}

        if is_local:
            for mode in self.mask_modes:
                masks[mode] = self(images, mode)
        else:
            masks['recon'] = self(images, 'global')
            masks['blend'] = torch.clone(masks['recon'])
            masks['dynamic'] = torch.clone(masks['recon'])
        return masks


if __name__ == '__main__':
    parser = FaceParser()

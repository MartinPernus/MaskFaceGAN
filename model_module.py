import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS

from models import BranchedTiny, FaceParser, Generator

from utils import interpolate

def batch_mse_loss(input, target, mask=None):
    loss = F.mse_loss(input, target, reduction='none')
    if mask is not None:
        loss = mask * loss
    loss_for_image = torch.mean(loss, dim=(1, 2, 3))
    return loss_for_image

class ModelsModule(nn.Module):
    def __init__(self, attribute_subset=None, update_shape=False, is_local=False, blend_option='include_skin'):
        super().__init__()
        self.update_shape = update_shape
        self.is_local = is_local

        self.classifier = BranchedTiny()
        self.face_parser = FaceParser(blend_option=blend_option)
        self.generator = Generator(1024, 512, 8)
        if not self.is_local:
            self.lpips = LPIPS()

        latent_mean = self.generator.get_latent_statistics()
        self.register_buffer('latent_mean', latent_mean)

        self.requires_grad_(False)
        self.eval()

        if attribute_subset is not None:
            self.set_attributes(attribute_subset)

    def recon_loss(self, img1, img2, mask):
        if self.is_local:
            return batch_mse_loss(img1, img2, mask=mask)
        else:
            return self.lpips(img1, img2, mask=mask)

    def set_attribute(self, attribute):
        self.classifier.set_idx(attribute)
        self.face_parser.set_idx(attribute)

    def set_attributes(self, attributes):
        if not isinstance(attributes, list):
            attributes = [attributes]

        self.classifier.set_idx_list(attributes)
        self.face_parser.set_idx_list(attributes)

    def get_noise(self, n, trainable=False, random=False):
        noises = self.generator.make_noise(n, randomize=random)
        noises = [x.requires_grad_(trainable) for x in noises]
        return noises

    def get_latent(self, n, trainable=False):
        latent = self.latent_mean.clone().repeat(n, 18, 1)
        latent.requires_grad_(trainable)
        return latent

    @torch.no_grad()
    def update_image_masks(self, images, source_masks, target_masks):
        dynamic_mask = self.face_parser(images, 'dynamic').sum(1, keepdim=True)

        target_masks['recon'] = source_masks['recon'] - dynamic_mask
        target_masks['dynamic'] = source_masks['dynamic'] + dynamic_mask
        if self.update_shape:
            target_masks['shape'] = source_masks['shape'] + interpolate(dynamic_mask, 512)

        for mode in target_masks:
            target_masks[mode].clamp_(0, 1)

        return target_masks

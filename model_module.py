import torch
import torch.nn as nn

from models import BranchedTiny, FaceParser, Generator

from utils import interpolate

class ModelsModule(nn.Module):
    def __init__(self, cfg, attribute_subset=None):
        super().__init__()
        self.cfg = cfg
        self.update_shape = cfg.SEGMENTATOR.UPDATE_SHAPE

        self.classifier = BranchedTiny()
        self.face_parser = FaceParser(cfg.SEGMENTATOR)
        self.generator = Generator(1024, 512, 8)

        latent_mean = self.generator.get_latent_statistics()
        self.register_buffer('latent_mean', latent_mean)

        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

        if attribute_subset is not None:
            self.set_attributes(attribute_subset)


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
    def get_image_masks(self, images):
        assert images.size(2) == images.size(3)
        masks = {}
        for mode in self.face_parser.mask_modes:
            masks[mode] = self.face_parser(images, mode)
        return masks

    @torch.no_grad()
    def update_image_masks(self, images, source_masks, target_masks):
        dynamic_mask = self.face_parser(images, 'dynamic').sum(1, keepdim=True)

        target_masks['mse'] = source_masks['mse'] - dynamic_mask
        target_masks['dynamic'] = source_masks['dynamic'] + dynamic_mask
        if self.update_shape:
            target_masks['shape'] = source_masks['shape'] + interpolate(dynamic_mask, 512)

        for mode in target_masks:
            target_masks[mode].clamp_(0, 1)

        return target_masks

    def add_noise(self, mask, thresh=0.95, n=1):

        noise_masks = self.get_noise_mask(mask, thresh)
        noises = self.get_noise(n, trainable=False)

        for i in range(len(noises)):
            noise_mask = noise_masks[i]
            n = torch.sum(noise_mask)
            noises[i][noise_mask] = torch.randn(n, device=noises[i].device)
        return noises

    def get_noise_mask(self, mask, thresh):
        mask = mask > thresh
        n = mask.size(0)

        latent = self.get_latent(n, trainable=False)
        noises = self.get_noise(n, trainable=True)
        img_gen = self.g(latent, noise=noises)
        loss = (img_gen * mask).sum()
        loss.backward()

        noise_grads = [x.grad for x in noises]
        noise_masks = [(x == 0) for x in noise_grads]  # which noise components do not affect the mask_mse
        return noise_masks


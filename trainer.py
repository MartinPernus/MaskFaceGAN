import copy

import torch
import torch.optim as optim

from loss_function_helpers import noise_normalize_, noise_regularize, kl_divergence
from model_module import batch_mse_loss

def e4e_latent_prediction(image):
    from models.e4e import Encoder
    enc = Encoder().to(image.device)
    latent = enc(image)
    return latent

class Trainer():
    def __init__(self, models, image, target, cfg):
        self.models = models
        self.image = image
        self.target = target
        self.cfg = cfg
        
        n = len(self.image)        
        self.noises = self.models.get_noise(n, trainable=False)
        self.masks = self.models.face_parser.get_image_masks(self.image, cfg.models.is_local)
        self.original_masks = copy.deepcopy(self.masks)

        if self.cfg.e4e_init:
            self.latent = e4e_latent_prediction(self.image)
            self.start_steps = 300
        else:
            self.latent = self.models.get_latent(n, trainable=False)
            self.start_steps = 0

    @property
    def device(self):
        return next(self.models.parameters()).device

    def train_latent(self):
        n = len(self.image)

        assert self.latent.size(0) == n
        assert all([x.size(0) == n for x in self.noises])

        n = self.latent.size(0)
        self.latent.requires_grad_(True)

        size_multiplier = torch.tensor([self.cfg.size]).view(1, -1).to(self.device)
        target_attr_portion = size_multiplier * get_component_portion(self.masks['dynamic'])

        optimizer = optim.Adam([self.latent], lr=self.cfg.optimizer.lr_latent)

        losses = []
        dynamic_masking_start = self.cfg.loss.start_steps.seg
        dynamic_masking_end = dynamic_masking_start + self.cfg.dynamic_masking_iters

        for i in range(self.start_steps, self.cfg.steps.latent):

            if self.cfg.optimizer.reinit and i == self.cfg.loss.start_steps.classf:
                optimizer = optim.Adam([self.latent], lr=self.cfg.optimizer.lr_latent)

            if i == dynamic_masking_end and self.cfg.steps.noise > 0:
                self.noises = self.models.get_noise(n, trainable=False, random=True)

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)

            loss = {}
            loss['recon'] = self.models.recon_loss(self.image, images_generated, self.masks['recon'])

            if self.cfg.loss.weights.classf > 0 and i >= self.cfg.loss.start_steps.classf:
                attr_pred = torch.sigmoid(self.models.classifier(images_generated))
                loss['classf'] = kl_divergence(attr_pred, self.target)

            if self.cfg.loss.weights.seg > 0 and i >= self.cfg.loss.start_steps.seg:
                i_dynamic_mask = self.models.face_parser(images_generated, mode='shape')
                loss['seg'] = batch_mse_loss(i_dynamic_mask, self.masks['shape'])

            if self.cfg.size > 0 and i >= self.cfg.loss.start_steps.seg:
                i_dynamic_mask = self.models.face_parser(images_generated, mode='dynamic')
                i_img_portion = get_component_portion(i_dynamic_mask)
                loss['size'] = kl_divergence(i_img_portion, target_attr_portion)

            if i % self.cfg.steps.log == 0:
                print(f'Latent optimization {i:03d}/{self.cfg.steps.latent}')



            loss_sum = 0
            for term in loss:
                weight = self.cfg.loss.weights[term]
                loss_sum += weight * loss[term].sum()

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()

            if self.cfg.dynamic_masking and dynamic_masking_start <= i < dynamic_masking_end:
                self.masks = self.models.update_image_masks(images_generated, self.original_masks, self.masks)

        print('Finished latent optimization.\n')
        return losses

    def train_noise(self):
        self.noises = [x.requires_grad_(True) for x in self.noises]
        self.latent = self.latent.detach()
        optimizer = optim.Adam(self.noises, lr=self.cfg.optimizer.lr_noise)

        for i in range(self.cfg.steps.noise+1):
            if i % self.cfg.steps.log == 0:
                print(f'Noise optimization {i:03d}/{self.cfg.steps.noise}')

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
            loss = {}
            loss['n_loss'] = noise_regularize(self.noises)
            noise_normalize_(self.noises)

            loss['recon'] = batch_mse_loss(self.image, images_generated, self.masks['recon'])
            loss_sum = 0
            for term in loss:
                loss_sum += self.cfg.loss.weights[term] * loss[term]

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Finished noise optimization.\n')


    @torch.no_grad()
    def generate_result(self):
        images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
        result = self.masks['blend'] * images_generated + (1 - self.masks['blend']) * self.image
        return result

    def generate_image(self):
        return self.models.generator(self.latent, noise=self.noises, to_01=True)

    @torch.no_grad()
    def generate_comparison(self):
        comparisons = []
        results = self.generate_result()
        for original, res in zip(self.image, results):
            comparisons.append(torch.stack((original, res)))
        comparisons = torch.cat(comparisons, dim=0)
        return comparisons

def get_component_portion(mask):
    return mask.sum(dim=(2,3)) / (mask.size(2) * mask.size(3))

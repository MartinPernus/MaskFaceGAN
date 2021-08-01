import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from loss_function_helpers import noise_normalize_, noise_regularize, kl_divergence

class Trainer():
    def __init__(self, image, models, target, cfg):
        self.models = models.to(cfg.DEVICE)
        self.image = image.to(cfg.DEVICE)
        self.target = target.to(cfg.DEVICE)

        n = len(image)

        if cfg.USE_E4E:
            self.latent = self.e4e_latent_prediction(self.image)
            self.start_steps = 300
        else:
            self.latent = models.get_latent(n, trainable=False)
            self.start_steps = 0

        self.noises = models.get_noise(n, trainable=False)

        assert self.latent.size(0) == n
        assert all([x.size(0) == n for x in self.noises])

        self.masks = models.get_image_masks(self.image)
        self.original_masks = copy.deepcopy(self.masks)
        self.cfg = cfg

    @property
    def device(self):
        return next(self.models.parameters()).device


    def e4e_latent_prediction(self, image):
        from models.e4e import Encoder
        enc = Encoder().to(image.device)
        latent = enc(image)
        return latent

    def train_latent(self):
        n = self.latent.size(0)
        self.latent.requires_grad_(True)

        size_multiplier = torch.tensor([self.cfg.SIZE]).view(1, -1).to(self.device)
        target_attr_portion = size_multiplier * get_component_portion(self.masks['dynamic'])

        optimizer = optim.Adam([self.latent], lr=self.cfg.OPTIMIZER.LR_LATENT)

        losses = []
        dynamic_masking_start = self.cfg.LOSS.start_steps.seg
        dynamic_masking_end = dynamic_masking_start + self.cfg.DYNAMIC_MASKING_ITERS

        for i in range(self.start_steps, self.cfg.N_LATENT_STEPS):
            if i % self.cfg.N_ITER_PRINT == 0:
                print(f'Latent optimization {i}/{self.cfg.N_LATENT_STEPS}')

            if self.cfg.OPTIMIZER.REINIT and i == self.cfg.LOSS.start_steps.classf:
                optimizer = optim.Adam([self.latent], lr=self.cfg.OPTIMIZER.LR_LATENT)

            if i == dynamic_masking_end and self.cfg.N_NOISE_STEPS > 0:
                self.noises = self.models.get_noise(n, trainable=False, random=True)

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)

            loss = {}
            loss['mse'] = batch_mse_loss(self.image, images_generated, self.masks['mse'])

            if self.cfg.LOSS.weights.classf > 0 and i >= self.cfg.LOSS.start_steps.classf:
                attr_pred = torch.sigmoid(self.models.classifier(images_generated))
                loss['classf'] = kl_divergence(attr_pred, self.target)

            if self.cfg.LOSS.weights.seg > 0 and i >= self.cfg.LOSS.start_steps.seg:
                i_dynamic_mask = self.models.face_parser(images_generated, mode='shape')
                loss['seg'] = batch_mse_loss(i_dynamic_mask, self.masks['shape'])

            if self.cfg.SIZE > 0 and i >= self.cfg.LOSS.start_steps.seg:
                i_dynamic_mask = self.models.face_parser(images_generated, mode='dynamic')
                i_img_portion = get_component_portion(i_dynamic_mask)
                loss['size'] = kl_divergence(i_img_portion, target_attr_portion)

            loss_sum = 0
            for term in loss:
                weight = self.cfg.LOSS.weights[term]
                loss_sum += weight * loss[term].sum()

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()


            if self.cfg.DYNAMIC_MASKING and dynamic_masking_start <= i < dynamic_masking_end:
                self.masks = self.models.update_image_masks(images_generated, self.original_masks, self.masks)

        print('Finished latent optimization.\n')
        return losses

    def train_noise(self):
        self.noises = [x.requires_grad_(True) for x in self.noises]
        self.latent = self.latent.detach()
        optimizer = optim.Adam(self.noises, lr=self.cfg.OPTIMIZER.LR_NOISE)

        for i in range(self.cfg.N_NOISE_STEPS+1):
            if i % self.cfg.N_ITER_PRINT == 0:
                print(f'Noise optimization {i}/{self.cfg.N_NOISE_STEPS}')

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
            loss = {}
            loss['n_loss'] = noise_regularize(self.noises)
            noise_normalize_(self.noises)

            loss['mse'] = batch_mse_loss(self.image, images_generated, self.masks['mse'])
            loss_sum = 0
            for term in loss:
                loss_sum += self.cfg.LOSS.weights[term] * loss[term]

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Finished noise optimization.\n')


    @torch.no_grad()
    def generate_result(self):
        images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
        blend_mask = self.models.face_parser(images_generated, 'blend')
        result = blend_mask * images_generated + (1 - blend_mask) * self.image
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


def batch_mse_loss(input, target, mask=None):
    loss = F.mse_loss(input, target, reduction='none')
    if mask is not None:
        loss = mask * loss
    loss_for_image = torch.mean(loss, dim=(1, 2, 3))
    return loss_for_image

def get_component_portion(mask):
    assert mask.size(2) * mask.size(3)  # todo: delete
    return mask.sum(dim=(2,3)) / (mask.size(2) * mask.size(3))

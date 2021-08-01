import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

from .psp_encoders import Encoder4Editing

class Encoder(nn.Module):
    def __init__(self, mode='w+'):
        super().__init__()
        self.mode = mode

        thisdir = Path(__file__).parent

        state = torch.load(thisdir / 'state.pt')
        opts = state['options']
        encoder_state_dict = state['encoder_state_dict']

        self.encoder = Encoder4Editing(50, 'ir_se', opts)
        self.encoder.load_state_dict(encoder_state_dict)

        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.img_size = 256
        self.eval()
        self.requires_grad_(False)

        self.register_buffer('latent_avg', state['latent_avg'])

    def downsample(self, image):
        return F.interpolate(image, (self.img_size, self.img_size), mode='area')

    def forward(self, image_01):
        image = (image_01 - self.mean) / self.std
        if image.size(-1) !=  self.img_size:
            image = self.downsample(image)

        codes = self.encoder(image, mode=self.mode)
        if codes.ndim == 2:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes


if __name__ == '__main__':
    enc = Encoder()

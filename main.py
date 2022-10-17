import os
from pathlib import Path
import torch
import argparse

from utils import save_image, read_img, Config
from trainer import Trainer
from model_module import ModelsModule
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=str, default='mouth_slightly_open')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--image', type=str, default='input/1815.jpg')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=0.05)
    parser.add_argument('--size', type=float, default=0)
    parser.add_argument('--e4e_init', action='store_true')
    parser.add_argument('--force_global', action='store_true')
    parser.add_argument('--blend_option', choices=['include_skin', 'no_skin'], default='include_skin')
    args = parser.parse_args()
    return args

def load_data(image_file, target, device='cuda:0', smoothing=0.05):
    image = read_img(image_file)
    target = torch.tensor(target).float().unsqueeze(0)
    target = torch.abs(target - smoothing)
    return image.to(device), target.to(device)

if __name__ == '__main__':
    args = parse_args()  # n, gpu
    cfg = Config('config.yml', attribute=args.attribute, size=args.size, e4e_init=args.e4e_init,
            blend_option=args.blend_option).cfg

    image, target = load_data(args.image, args.target, 
                            device=cfg.device, smoothing=args.smoothing)

    models = ModelsModule(attribute_subset=cfg.attributes, update_shape=cfg.models.update_shape,
                        is_local=cfg.models.is_local, blend_option=cfg.blend_option).to(cfg.device)
    trainer = Trainer(models, image, target, cfg)

    trainer.train_latent()
    trainer.train_noise()
    img_result = trainer.generate_result()

    os.makedirs(args.outdir, exist_ok=True)
    save_image(img_result, os.path.join(args.outdir, f'{Path(args.image).stem}-{args.attribute}.jpg'))
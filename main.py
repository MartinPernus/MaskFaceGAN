import os
#os.environ['PATH'] += '/usr/local/cuda/bin:/usr/local/bin/:/usr/bin:'
#os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/include:/usr/local/lib/"
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model_module import ModelsModule

from utils import *
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=None)
    parser.add_argument('--attribute', type=str, default='mouth_slightly_open')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--image', type=str, default='indir/1815.jpg')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=0.05)
    parser.add_argument('--use_e4e', type=bool, default=False)
    parser.add_argument('--size', type=float, default=0)
    args = parser.parse_args()
    return args


def load_data(image_file, target, smoothing=0.05):
    image = read_img(image_file)
    image_name = Path(args.image).name
    target = torch.tensor(target).float().unsqueeze(0)
    target = torch.abs(target - smoothing)
    return image, image_name, target

def save_results(img_result, image_name, outdir='.'):
    os.makedirs(outdir, exist_ok=True)
    save_image(img_result, os.path.join(outdir, image_name))

if __name__ == '__main__':
    args = parse_args()  # n, gpu
    cfg = load_config('config.yml', args)

    image, image_name, target = load_data(args.image, args.target, smoothing=args.smoothing)
    models = ModelsModule(cfg.MODELS, attribute_subset=cfg.ATTRIBUTES).to(cfg.DEVICE)

    trainer = Trainer(image, models, target, cfg)
    trainer.train_latent()
    trainer.train_noise()
    img_result = trainer.generate_result()

    save_results(img_result, image_name, outdir=cfg.OUTDIR)



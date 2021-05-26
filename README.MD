# High Resolution Face Editing with Masked GAN Latent Code Optimization

[Paper](https://arxiv.org/abs/2103.11135)

We propose a method for high resolution face editing through the use of constraints on GAN inpainted image regions.

## Setup

### 1. Install requirements
pip install -r requirements.txt

### 2. Download pretrained models

#### StyleGAN2
Convert the official weights, by cloning and following instructions of 
[rosinality's PyTorch StyleGAN implementation ](https://github.com/rosinality/stylegan2-pytorch).
Copy the created `stylegan2-ffhq-config-f.pt` checkpoint to models/stylegan2 folder.

#### Other models (attribute classifier and face parser)
Download the model checkpoint with download.sh script

`./download.sh`

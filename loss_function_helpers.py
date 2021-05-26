import torch
import torch.nn.functional as F

def noise_regularize(noises):
    loss = 0
    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean((1,2,3)).pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean((1,2,3)).pow(2)
            )
            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def fft_regularize(noises):  # does not work yet ...
    loss = torch.zeros(noises[0].size(0))

    for noise in noises:
        F_ = torch.fft.fftn(noise, 2)
        S = F_ * torch.conj(F_)
        R = torch.fft.ifftn(S, 2)
        loss += torch.mean(F.mse_loss(R.abs(), torch.zeros_like(R), reduction='none'), dim=(1,2,3))

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean(dim=(1,2,3), keepdim=True)
        std = noise.std(dim=(1,2,3), keepdim=True)

        noise.data.add_(-mean).div_(std)

def to_distribution(prob):
    distribution = prob.unsqueeze(-1)
    distribution = torch.cat((distribution, (1-distribution)), dim=-1)
    distribution = distribution.float()
    return distribution

def kl_divergence(input, target):
    input = to_distribution(input)
    target = to_distribution(target)
    kl = F.kl_div(input.log(), target, reduction='none')
    kl = torch.sum(kl, dim=-1)
    if kl.ndim > 1:
        kl = torch.mean(kl, dim=1)
    return kl

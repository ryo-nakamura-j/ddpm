import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import UNET, SinusoidalEmbeddings, ResBlock
from diffusion.sampler import DDPM_Scheduler
from train import train
from einops import rearrange
from typing import List
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
import matplotlib.animation as animation
from IPython.display import display, HTML
import numpy as np
import math

def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()

def animate(images):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis('off')

    ims = []
    for idx, img in enumerate(images):
        # bring into HWC numpy array
        img_np = rearrange(img.squeeze(0), 'c h w -> h w c').detach().numpy()

        # compute pre-normalization range
        min_pre, max_pre = img_np.min(), img_np.max()

        # normalize into [0,1]
        img_np = (img_np - min_pre) / (max_pre - min_pre + 1e-8)

        # compute post-normalization range
        min_post, max_post = img_np.min(), img_np.max()

        # print them
        print(f"Frame {idx:3d}: pre-range [{min_pre:.4f}, {max_pre:.4f}], "
              f"post-range [{min_post:.4f}, {max_post:.4f}]")

        # add to animation
        im = ax.imshow(img_np, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    plt.close(fig)
    return HTML(ani.to_html5_video())

def animate_grid(runs: List[List], n_rows: int = None, n_cols: int = None, interval: int = 100):
    """
    runs: a list of lists of tensors, each inner list comes from one call to inference()
    n_rows/n_cols: layout of the grid (auto‐computed if None)
    """
    num_runs = len(runs)
    num_frames = len(runs[0])

    # choose grid shape
    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(num_runs)))
    if n_rows is None:
        n_rows = int(math.ceil(num_runs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.8 * n_cols, 1.8 * n_rows))
    plt.tight_layout(pad=0)
    # flatten in case axes is 2D or 1D
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    ims = []
    for t in range(num_frames):
        artists = []
        for i, run in enumerate(runs):
            ax = axes_flat[i]
            ax.axis('off')
            img_np = rearrange(run[t].squeeze(0), 'c h w -> h w c').detach().numpy()
            # normalize into [0,1]
            mn, mx = img_np.min(), img_np.max()
            img_np = (img_np - mn) / (mx - mn + 1e-8)
            artists.append(ax.imshow(img_np, animated=True))
        ims.append(artists)

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    plt.close(fig)
    return HTML(ani.to_html5_video())

def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999, ):
    checkpoint = torch.load(checkpoint_path)
    model = UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    # times = [0,15,50,100,200,300,400,550,700,999]
    capture_steps = list(range(0, 10)) + list(range(10, 200, 20)) + list(range(200, 400, 50)) + list(range(400, 900, 100)) + [999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        # z = torch.randn(1, 3, 176, 176)
        z = torch.randn(1, 1, 32, 32)
        for t in reversed(range(1, num_time_steps)):
            t = [t]
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
            z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(),t).cpu())
            if t[0] in capture_steps:
                images.append(z.clone())
            # e = torch.randn(1, 3, 176, 176)
            e = torch.randn(1, 1, 32, 32)
            z = z + (e*torch.sqrt(scheduler.beta[t]))
        temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
        x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),[0]).cpu())

        images.append(x.clone()) 
    return images

def main():
    inference(checkpoint_path='Models/checkpoints/celeba_epoch_100.pth')

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import UNET
from diffusion.sampler import DDPM_Scheduler    
import random
import torch.optim as optim
from timm.utils import ModelEmaV3
from tqdm import tqdm
from torchvision.datasets import CelebA
from torch.cuda.amp import autocast, GradScaler
import argparse
import yaml
import os
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(batch_size: int,
          num_time_steps: int,
          num_epochs: int,
          seed: int,
          ema_decay: float,
          lr: float,
          checkpoint_dir: str,
          checkpoint_prefix: str,
          save_every: int,
          log_dir: str,
          checkpoint_path: str=None):

    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    # CelebA → 218×178 → square 176×176 → no resize
    image_size = 176
    tfm = transforms.Compose([
        transforms.CenterCrop(178),           # 218×178 → 178×178
        transforms.CenterCrop(image_size),    # 178×178 → 176×176
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_dataset = CelebA(root='./Data',
                           split='train', download=False,
                           transform=tfm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET(input_channels=3,
                 output_channels=3,
                 time_steps=num_time_steps).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    # prepare logging & checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    scaler = GradScaler()

    global_step = 0
    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.cuda()
            # x = F.pad(x, (2,2,2,2))
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            optimizer.zero_grad()
            with autocast():
                output = model(x, t)
                loss = criterion(output, e)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.item()
            global_step += 1
            writer.add_scalar('loss', loss.item(), global_step)

        avg_loss = total_loss / (len(train_loader))
        writer.add_scalar('loss', avg_loss, global_step)
        print(f'Epoch {i+1} | Loss {avg_loss:.5f}')

        if (i+1) % save_every == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, f'{checkpoint_dir}/{checkpoint_prefix}_epoch_{i+1}.pth')

    # final save
    final_ckpt = {
        'weights':   model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema':       ema.state_dict()
    }
    torch.save(final_ckpt,
               os.path.join(checkpoint_dir, f"{checkpoint_prefix}_final.pt"))
    writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='Configs/config.yaml')
    args = p.parse_args()

    # load hyperparameters from YAML
    cfg = yaml.safe_load(open(args.config))['train']
    train(**cfg)

if __name__ == '__main__':
    main()
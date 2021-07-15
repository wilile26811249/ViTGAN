import os
import argparse
import copy
from tqdm import tqdm
import numpy as np

import utils
import models

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid


def exp_mov_avg(Gs, G, alpha = 0.999, global_step = 999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(generator, generator_s, discriminator, optim_g, optim_d, data_loader, device):
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim))).to(device)
    for step in tqdm(range(args.steps + 1)):
        # Train Discriminator
        optim_d.zero_grad()

        # Forward + Backward with real images
        r_img = next(data_loader).to(device)
        r_label = torch.ones(args.batch_size).to(device)
        r_logit = discriminator(r_img).flatten()
        lossD_real = criterion(r_logit, r_label)
        lossD_real.backward()

        # Forward + Backward with fake images
        latent_vector = torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(device)
        f_img = generator(latent_vector)
        f_label = torch.zeros(args.batch_size).to(device)
        f_logit = discriminator(f_img).flatten()
        lossD_fake = criterion(f_logit, f_label)
        lossD_fake.backward()

        optim_d.step()

        # Train Generator
        optim_g.zero_grad()
        f_img = generator(torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(device))
        r_label = torch.ones(args.batch_size).to(device)
        f_logit = discriminator(f_img).flatten()
        lossG = criterion(f_logit, r_label)
        lossG.backward()
        optim_g.step()

        exp_mov_avg(generator_s, generator, global_step = step)

        if step % args.sample_interval == 0:
            generator.eval()
            vis = generator(fixed_noise).detach().cpu()
            vis = make_grid(vis, nrow = 4, padding = 5, normalize = True)
            vis = T.ToPILImage()(vis)
            vis.save('samples/vis{:05d}.jpg'.format(step))
            generator.train()
            print("Save sample to samples/vis{:05d}.jpg".format(step))

        if (step + 1) % args.sample_interval == 0 or step == 0:
            # Save the checkpoints.
            torch.save(generator.state_dict(), 'weights/Generator.pth')
            torch.save(generator_s.state_dict(), 'weights/Generator_ema.pth')
            torch.save(discriminator.state_dict(), 'weights/Discriminator.pth')
            print("Save model state.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 100000,
                        help = "Number of steps for training (Default: 100000)")
    parser.add_argument("--batch-size", type = int, default = 128,
                        help = "Size of each batches (Default: 128)")
    parser.add_argument("--lr", type = float, default = 0.002,
                        help = "Learning rate (Default: 0.002)")
    parser.add_argument("--beta1", type = float, default = 0.0,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--latent-dim", type = int, default = 1024,
                        help = "Dimension of the latent vector")
    parser.add_argument("--data-dir", type = str, default = "crypko_data/faces/",
                        help = "Data root dir of your training data")
    parser.add_argument("--sample-interval", type = int, default = 1000,
                        help = "Interval for sampling image from generator")
    parser.add_argument("--gpu-id", type = int, default = 1,
                        help = "Select the specific gpu to training")
    args = parser.parse_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    data_loader = utils.get_dataloader(args.data_dir, batch_size = args.batch_size)


    # Create the log folder
    os.makedirs("weights", exist_ok = True)
    os.makedirs("samples", exist_ok = True)

    # Initialize Generator and Discriminator
    netG = models.Generator().to(device)
    netG_s = copy.deepcopy(netG)
    netD = models.Discriminator().to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer and lr_scheduler
    optimizer_g = torch.optim.Adam(netG.parameters(), lr = args.lr,
        betas = (args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(netD.parameters(), lr = args.lr,
        betas = (args.beta1, args.beta2)
    )

    # Start Training
    train(netG_s, netG, netD, optimizer_g, optimizer_d, data_loader, device)

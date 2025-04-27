import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="cifar-10-batches-py",
        help="path to folder containing data_batch_1…data_batch_5 and test_batch"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cifar 10 result",
        help="where to save .png and .npz outputs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--epochs",     type=int, default=30,  help="number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimension of VAE latent space")
    parser.add_argument("--lr",         type=float, default=2e-4, help="learning rate")
    return parser.parse_args()

class CIFAR10Pickle(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        root_dir should contain:
          data_batch_1 … data_batch_5 and test_batch
        """
        self.transform = transform
        batch_names = [f"data_batch_{i}" for i in range(1,6)] if train else ["test_batch"]
        data_list, label_list = [], []
        for fname in batch_names:
            path = os.path.join(root_dir, fname)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            data_list.append(batch[b"data"])
            label_list += batch[b"labels"]
        data = np.concatenate(data_list, axis=0)
        data = data.reshape(-1, 3, 32, 32)
        self.images = data
        self.labels = label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.transpose(img, (1, 2, 0))  # to HWC
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 3×64×64 → 512×4×4
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.fc_mu     = nn.Linear(512*4*4, latent_dim)
        self.fc_logvar = nn.Linear(512*4*4, latent_dim)

        # Decoder: latent → 512×4×4 → 3×64×64
        self.decoder_input = nn.Linear(latent_dim, 512*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64,  3, 4, 2, 1),                  nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std  = torch.exp(0.5 * logvar)
        eps  = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z).view(-1, 512, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon = F.mse_loss(recon_x, x, reduction="sum")
    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

def main():
    multiprocessing.freeze_support()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(64),                      # 64×64 input
        transforms.ToTensor(),                      # [0,1]
        transforms.Normalize((0.5,0.5,0.5),         # [–1,1]
                             (0.5,0.5,0.5))
    ])

    # DataLoader
    dataset    = CIFAR10Pickle(root_dir=args.data_dir, train=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ConvVAE(latent_dim=args.latent_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:02d}/{args.epochs}  Avg Loss: {avg_loss:.4f}")

        # Save reconstructions every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample, _, _ = model(imgs[:16])
                grid = vutils.make_grid((sample + 1) / 2, nrow=4, padding=2)
                png_path = os.path.join(args.output_dir, f"recon_epoch{epoch}.png")
                vutils.save_image(grid, png_path)
                npz_path = os.path.join(args.output_dir, f"recon_epoch{epoch}.npz")
                np.savez_compressed(npz_path, recon=sample.cpu().numpy())
            model.train()

if __name__ == "__main__":
    main()
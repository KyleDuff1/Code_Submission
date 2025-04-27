import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from PIL import Image
import multiprocessing

# ─── argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="img_align_celeba",
        help="path to folder containing all CelebA .jpg files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cifar a result",
        help="where to save .png and .npz outputs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--epochs",     type=int, default=30,  help="number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimension of VAE latent space")
    parser.add_argument("--lr",         type=float, default=2e-4, help="learning rate")
    return parser.parse_args()

# ─── custom CelebA folder dataset ──────────────────────────────────────────────
class CelebADir(Dataset):
    """Loads all .jpg images from a single directory."""
    def __init__(self, root_dir, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

# ─── VAE definition (same as before) ───────────────────────────────────────────
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.fc_mu     = nn.Linear(512*4*4, latent_dim)
        self.fc_logvar = nn.Linear(512*4*4, latent_dim)
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

# ─── main training loop ────────────────────────────────────────────────────────
def main():
    multiprocessing.freeze_support()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # transforms: center‑crop, resize to 64×64, normalize to [–1,1]
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # replace CIFAR loader with CelebA loader
    dataset    = CelebADir(root_dir=args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.999))

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:02d}/{args.epochs}  Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample, _, _ = model(imgs[:16])
                grid = vutils.make_grid((sample+1)/2, nrow=4, padding=2)
                png_path = os.path.join(args.output_dir, f"recon_celebA_epoch{epoch}.png")
                vutils.save_image(grid, png_path)
                npz_path = os.path.join(args.output_dir, f"recon_celebA_epoch{epoch}.npz")
                np.savez_compressed(npz_path, recon=sample.cpu().numpy())
            model.train()

if __name__ == "__main__":
    main()
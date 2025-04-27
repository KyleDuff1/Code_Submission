# metrics_cifara_npz.py

import os
import glob
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, PILToTensor
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms

class ReconFloatDataset(Dataset):
    """Loads all reconstructions from .npz, returns float32 [0,1], resized to 299×299."""
    def __init__(self, npz_folder):
        imgs = []
        for path in sorted(glob.glob(os.path.join(npz_folder, "*.npz"))):
            arr = np.load(path)["recon"]            # shape (B,3,64,64), float in [-1,1]
            arr = (arr + 1.0) / 2.0                 # to [0,1]
            imgs.append(torch.from_numpy(arr))      # Tensor(B,3,64,64)
        self.images = torch.cat(imgs, dim=0)       # Tensor(N,3,64,64)
        self.resize = Resize((299, 299))

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        img = self.images[idx]                    # float32 [0,1]
        img = self.resize(img)                    # float32 [0,1], 3×299×299
        return img

class ReconUInt8Dataset(Dataset):
    """Loads same .npz but converts to uint8 [0–255], resized to 299×299."""
    def __init__(self, npz_folder):
        imgs = []
        for path in sorted(glob.glob(os.path.join(npz_folder, "*.npz"))):
            arr = np.load(path)["recon"]            # float in [-1,1]
            arr = ((arr + 1.0) / 2.0 * 255.0).astype(np.uint8)
            imgs.append(torch.from_numpy(arr))      # uint8 Tensor(B,3,64,64)
        self.images = torch.cat(imgs, dim=0)       # uint8 Tensor(N,3,64,64)
        self.resize = Resize((299, 299))

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        img = self.images[idx]                    # uint8 [0,255]
        img = self.resize(img)                    # uint8 [0,255], 3×299×299
        return img

def compute_is_fid(gen_folder, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Float dataset for IS
    float_ds = ReconFloatDataset(gen_folder)
    float_loader = DataLoader(float_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # UInt8 dataset for FID
    uint_ds = ReconUInt8Dataset(gen_folder)
    uint_loader = DataLoader(uint_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Real CIFAR-10 for FID (uint8)
    real_loader = DataLoader(
        datasets.CIFAR10(
            root=".", train=False, download=True,
            transform=transforms.Compose([
                Resize((299,299)),
                PILToTensor()               # yields uint8 [0,255]
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Inception Score (normalize=True expects float [0,1])
    is_metric = InceptionScore(normalize=True).to(device)
    for batch in float_loader:
        is_metric.update(batch.to(device))
    mean_is, std_is = is_metric.compute()
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")

    # FID (expects uint8)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    # real images first
    for real_imgs, _ in real_loader:
        fid_metric.update(real_imgs.to(device), real=True)
    # then generated
    for gen_imgs in uint_loader:
        fid_metric.update(gen_imgs.to(device), real=False)
    fid_score = fid_metric.compute()
    print(f"FID Score:       {fid_score:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gen_folder",
        type=str,
        default="cifar a result",
        help="folder containing your recon_epoch*.npz files"
    )
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    compute_is_fid(
        gen_folder=args.gen_folder,
        batch_size=args.batch_size
    )
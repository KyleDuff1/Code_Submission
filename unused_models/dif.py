import torch
import torch.nn as nn
import torchvision.utils as utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from pytorch_gan_metrics import get_inception_score, get_fid
import os

train_mode = False
epochs = 5
steps=1000

class Schedule():
    def __init__(self, sigmas):
        self.sigmas = sigmas

    def __getitem__(self, i):
        return self.sigmas[i]
    
    def sample_batch(self, x0):
        batch = x0.shape[0]
        return torch.randint(len(self.sigmas), (batch,), device=x0.device)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self):
        return 0

def training(loader, model, schedule):
    optimizer = torch.optim.Adam(model.parameters())
    device = next(model.parameters()).device

    for i in range(epochs):
        for x_0, j in loader:
            x_0 = x_0.to(device)
            optimizer.zero_grad()
            sigma = schedule.sample_batch(x_0)
            eps = torch.randn_like(x_0)

            eps_hat = model(x_0 + sigma * eps, sigma)

            loss = nn.MSELoss()(eps_hat, eps)

            loss.backward()
            optimizer.step()

def data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    return DataLoader(set, batch_size=128, shuffle=True)

def sample(model, schedule, num_samples):
    model.eval()

    device = next(model.parameters()).device
    x = torch.randn(num_samples, 3, 32, 32).to(device)

    for t in (range(steps)):
        sigma = schedule[steps - t - 1]
        sigma = sigma.expand(num_samples, 1, 1, 1).to(device)

        with torch.no_grad():
            noise_pred = model(x, sigma)

        x = x - sigma * noise_pred

    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)

    grid = utils.make_grid(x, nrow=4)
    utils.save_image(grid, "samples/sample.png")
    print("Sample image saved to samples/sample.png")

def evaluate_generated_images(generator, fixed_noise, output_dir="samples", epoch=0, stats_path='path/to/statistics.npz'):

    with torch.no_grad():
        x = fixed_noise.clone().to(next(generator.parameters()).device)

        for t in range(steps):
            sigma = schedule[steps - t - 1]
            sigma = sigma.expand(x.size(0), 1, 1, 1).to(x.device)
            eps_hat = generator(x, sigma)
            x = x - sigma * eps_hat

        fake_images = x.detach().cpu()

    composite_filename = os.path.join(output_dir, f"gan_epoch_{epoch:03d}.png")
    utils.save_image(fake_images, composite_filename, normalize=True)
    print(f"Saved composite generated images to {composite_filename}")

    fake_images_01 = (fake_images + 1) / 2.0
    fake_images_01 = torch.clamp(fake_images_01, 0, 1)

    try:
        IS, IS_std = get_inception_score(fake_images_01, batch_size=32, resize=True, splits=10)
        print(f"Inception Score: {IS:.2f} ± {IS_std:.2f}")
    except Exception as e:
        print(f"Error computing Inception Score: {e}")

    if not os.path.exists(stats_path):
        print(f"Statistics file not found at {stats_path}. Cannot compute FID.")
    else:
        try:
            FID = get_fid(fake_images_01, stats_path)
            print(f"FID: {FID:.2f}")
        except Exception as e:
            print(f"Error computing FID: {e}")





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

loader = data_loader()
model = CNN().to(device)
schedule = Schedule(sigmas=torch.linspace(0.01, 1.0, 1000).to(device))

if train_mode:
    training(loader, model, schedule)
    torch.save(model.state_dict(), "model.pth")
else:
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

sample(model, schedule, 16)

dataset = data_loader()
denoise_subset = torch.utils.data.Subset(dataset, range(1000))

x_clean, _ = next(iter(loader))
x_clean = x_clean[:8].to(device)

sigma = torch.tensor(0.2).to(device)
eps = torch.randn_like(x_clean)
x_noisy = x_clean + sigma * eps

model.eval()
with torch.no_grad():
    cnn_eps_hat = model(x_noisy, sigma.expand(8, 1, 1, 1))
    x_denoised_cnn = x_noisy - sigma * cnn_eps_hat



utils.save_image(x_clean, f"samples/clean.png", nrow=8)
utils.save_image(x_noisy, f"samples/noisy.png", nrow=8)
utils.save_image(x_denoised_cnn, f"samples/denoised.png", nrow=8)
print("Denoising comparison saved to samples/compare_denoisers.png")

fixed_noise = torch.randn(64, 3, 32, 32).to(device)

# evaluate_generated_images(
#     generator=model,
#     fixed_noise=fixed_noise,
#     output_dir="samples",
#     epoch=0,
#     stats_path="stats/cifar10.npz"
# )
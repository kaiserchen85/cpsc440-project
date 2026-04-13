import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vae import VAE
from utils import handle, main, run, load_dataset


def cvae_loss(x, x_hat, mu, logvar, beta=0.1, l1_weight=1.0, l2_weight=1.0):
    # Combined Reconstruction Loss (Sharpness + Overall Fit)
    l1_loss = F.l1_loss(x_hat, x)
    l2_loss = F.mse_loss(x_hat, x)
    recon_loss = (l1_weight * l1_loss) + (l2_weight * l2_loss)

    # KL Divergence
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss, kl_loss, recon_loss + (beta * kl_loss)



# TODO: Add vocoding step using API (Use HiFi-GAN)
# TODO: Make a function that can take in the actual dataset
@handle("vae-test")
def vae_test():
    # 1. Setup Dummy Data (B, C, H, W)
    x = torch.zeros(1, 1, 80, 200)

    # add diagonal line pattern
    for i in range(80):
        for j in range(200):
            if j % 20 == i % 20:
                x[0, 0, i, j] = 1.0
                
    text = torch.randint(0, 10000, (1, 10))
    label = torch.tensor([1])

    # 2. Initialize Model with exact shape
    model = VAE(spec_shape=(1, 80, 200))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training Loop with Beta Annealing
    total_steps = 200
    target_beta = 0.1

    for step in range(total_steps):
        # Calculate current beta (linear warmup over the first 100 steps)
        current_beta = target_beta * min(1.0, step / 100.0)

        x_hat, mu, logvar = model(x, text, label)

        recon, kl, total_loss = cvae_loss(x, x_hat, mu, logvar, beta=current_beta)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:3d} | Total: {total_loss.item():.4f} | "
                  f"Recon: {recon.item():.4f} | KL: {kl.item():.4f} | Beta: {current_beta:.4f}")

    # 4. Visualize
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0].detach().numpy(), aspect='auto', origin='lower')
    plt.title("Input")

    plt.subplot(1, 2, 2)
    plt.imshow(x_hat[0, 0].detach().numpy(), aspect='auto', origin='lower')
    plt.title("Reconstruction")
    plt.show()


################################################################################

if __name__ == "__main__":
    main()
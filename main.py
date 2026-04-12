import numpy as np
import torch
from tqdm import tqdm

from vae import VAE
from utils import handle, main, run, load_dataset



@handle("VAE")
def vae():
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for spectrogram, text, labels in dataloader:

        x_hat, mu, logvar = model(spectrogram, text, labels)

        loss = model.vae_loss(spectrogram, x_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # encode input (could be sincere)
    mu, logvar = model.encode(x_input)
    z = model.reparameterize(mu, logvar)

    # sample multiple variations
    # TODO: Add vocoder (usa API)
    outputs = []
    for _ in range(5):
        z_sample = z + 0.1 * torch.randn_like(z)

        sarcastic_label = torch.ones_like(label_input)  # force sarcasm

        x_gen = model.decode(z_sample, text_input, sarcastic_label)
        outputs.append(x_gen)










################################################################################

if __name__ == "__main__":
    main()
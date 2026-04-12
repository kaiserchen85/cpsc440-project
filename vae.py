import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        text_vocab_size=10000,
        text_embed_dim=128,
        num_classes=2  # sarcastic / sincere
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # CNN Encoder (audio only)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.fc_mu = None
        self.fc_logvar = None

        # Text embedding (decoder only)
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.text_fc = nn.Linear(text_embed_dim, 128)

        # Label embedding (decoder only)
        self.label_fc = nn.Linear(num_classes, 32)

        # Decoder (initialized later)
        self.fc_decode = None
        self.deconv = None

    # Initialize encoder FC
    def _init_encoder_fc(self, spec_shape):
        with torch.no_grad():
            dummy = torch.zeros(spec_shape)
            h = self.cnn_encoder(dummy)
            h = self.flatten(h)
            dim = h.shape[1]

        self.fc_mu = nn.Linear(dim, self.latent_dim)
        self.fc_logvar = nn.Linear(dim, self.latent_dim)

        return dim

    # Initialize decoder
    def _init_decoder(self, spec_shape, encoder_dim):
        # We reconstruct back to CNN feature map shape
        with torch.no_grad():
            dummy = torch.zeros(spec_shape)
            h = self.cnn_encoder(dummy)
            self.enc_shape = h.shape[1:]  # (C, H, W)
            flat_dim = h.numel() // h.shape[0]

        total_input = self.latent_dim + 128 + 32

        self.fc_decode = nn.Linear(total_input, flat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    # Encoder
    def encode(self, spectrogram):
        if self.fc_mu is None:
            enc_dim = self._init_encoder_fc(spectrogram.shape)
            self._init_decoder(spectrogram.shape, enc_dim)

        h = self.cnn_encoder(spectrogram)
        h = self.flatten(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    # Reparameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decode
    def decode(self, z, text_tokens, labels):
        # Text
        text_emb = self.text_embedding(text_tokens)     # (B, T, D)
        text_emb = text_emb.mean(dim=1)                 # simple pooling
        h_text = F.relu(self.text_fc(text_emb))         # (B, 128)

        # Label
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        h_label = F.relu(self.label_fc(labels_onehot))  # (B, 32)

        # Combine
        h = torch.cat([z, h_text, h_label], dim=1)

        h = self.fc_decode(h)

        # reshape to CNN feature map
        B = z.shape[0]
        C, H, W = self.enc_shape
        h = h.view(B, C, H, W)

        x_hat = self.deconv(h)

        return x_hat

    # Forward
    def forward(self, spectrogram, text_tokens, labels):
        mu, logvar = self.encode(spectrogram)
        z = self.reparameterize(mu, logvar)

        x_hat = self.decode(z, text_tokens, labels)

        return x_hat, mu, logvar
    
    
    def vae_loss(x, x_hat, mu, logvar, beta=0.1):
        recon_loss = F.mse_loss(x_hat, x)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + beta * kl_loss
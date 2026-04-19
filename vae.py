"""Convolutional conditional VAE on single-channel log-mel; see docs/MODEL.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        spec_shape=(1, 80, 130),  # Fallback; callers should pass dataset-derived (C, mel_bins, time)
        latent_dim=64, # number of latent variables to use
        text_vocab_size=10000, # size of vocabulary, TODO: Discuss about keep or not
        text_embed_dim=128, # mapping from vocab to integer, represents similarity
        num_classes=2  # 0: sincere, 1: sarcastic
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.spec_shape = spec_shape

        # 1. Text & Label Embeddings (Shared by Encoder and Decoder)
        # Translates words into mathematical values for matrix multiplication
        # 128 and 32 are hyperparameters
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.text_fc = nn.Linear(text_embed_dim, 128)
        self.label_fc = nn.Linear(num_classes, 32)

        # 2. CNN Encoder (Extracts audio features)
        # 3 layers with ReLU
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(spec_shape[0], 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        # Calculate flattened CNN shape dynamically during init by passing dummy spectrogram in
        with torch.no_grad():
            dummy = torch.zeros(1, *spec_shape)
            cnn_out = self.cnn_encoder(dummy)
            self.enc_shape = cnn_out.shape[1:]  # Save (C, H, W) for decoder
            cnn_flat_dim = cnn_out.numel() // cnn_out.shape[0]

        # 3. Latent Projections (Conditioned on Audio + Text + Tone)
        # We add the text (128) and label (32) dimensions to the CNN output
        total_enc_input = cnn_flat_dim + 128 + 32
        self.fc_mu = nn.Linear(total_enc_input, latent_dim)
        self.fc_logvar = nn.Linear(total_enc_input, latent_dim)

        # 4. Decoder Initialization
        total_dec_input = latent_dim + 128 + 32
        self.fc_decode = nn.Linear(total_dec_input, cnn_flat_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, spec_shape[0], 4, 2, 1),
            # Note: Depending on your mel-spec normalization, 
            # you might need a Tanh or no activation here.
        )

    def get_conditioning(self, text_tokens, labels):
        """Helper to process text and labels into dense vectors."""
        # Text: Pool over the sequence dimension (B, T) -> (B, 128)
        text_emb = self.text_embedding(text_tokens).mean(dim=1) 
        h_text = F.relu(self.text_fc(text_emb))                 

        # Labels: One-hot encode then project (B) -> (B, 32)
        # OHE used for ensuring network doesn't think sarcastic/sincere is ordinal in some way
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        h_label = F.relu(self.label_fc(labels_onehot))          
        
        return h_text, h_label

    def encode(self, spectrogram, text_tokens, labels):
        # Process audio
        h_audio = self.cnn_encoder(spectrogram)
        h_audio = self.flatten(h_audio)

        # Process conditions
        h_text, h_label = self.get_conditioning(text_tokens, labels)

        # Concatenate audio features with text and label conditions
        h_combined = torch.cat([h_audio, h_text, h_label], dim=1)

        # Compacts data
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, text_tokens, labels):
        # Process conditions (using the target labels during inference!)
        h_text, h_label = self.get_conditioning(text_tokens, labels)

        # Concatenate latent vector with conditions and decompress
        h_combined = torch.cat([z, h_text, h_label], dim=1)
        h_decoded = self.fc_decode(h_combined)

        # Reshape to match the feature map expected by ConvTranspose2d
        B = z.shape[0]
        h_reshaped = h_decoded.view(B, *self.enc_shape)

        # Decoding the data (ConvTranspose stack may be off by a few frames vs input width)
        x_hat = self.deconv(h_reshaped)
        if x_hat.shape[-2:] != self.spec_shape[1:]:
            x_hat = F.interpolate(
                x_hat, size=self.spec_shape[1:], mode="bilinear", align_corners=False
            )
        return x_hat

    def forward(self, spectrogram, text_tokens, labels):
        # 1. Encode with SOURCE labels
        mu, logvar = self.encode(spectrogram, text_tokens, labels)
        
        # 2. Sample
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode with SOURCE labels (during training to reconstruct)
        x_hat = self.decode(z, text_tokens, labels)

        return x_hat, mu, logvar
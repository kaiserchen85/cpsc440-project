import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
        self,
        spec_shape=(1, 80, 128), # (Channels, Mel-bins, Time-frames)
        latent_dim=64,
        text_vocab_size=10000,
        text_embed_dim=128,
        num_classes=2  # 0: sincere, 1: sarcastic
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.spec_shape = spec_shape

        # 1. Text & Label Embeddings (Shared by Encoder and Decoder)
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.text_fc = nn.Linear(text_embed_dim, 128)
        self.label_fc = nn.Linear(num_classes, 32)

        # 2. CNN Encoder (Extracts audio features)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(spec_shape[0], 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        # Calculate flattened CNN shape dynamically during init
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

        # Concatenate latent vector with conditions
        h_combined = torch.cat([z, h_text, h_label], dim=1)
        h_decoded = self.fc_decode(h_combined)

        # Reshape to match the feature map expected by ConvTranspose2d
        B = z.shape[0]
        h_reshaped = h_decoded.view(B, *self.enc_shape)

        x_hat = self.deconv(h_reshaped)
        return x_hat

    def forward(self, spectrogram, text_tokens, labels):
        # 1. Encode with SOURCE labels
        mu, logvar = self.encode(spectrogram, text_tokens, labels)
        
        # 2. Sample
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode with SOURCE labels (during training to reconstruct)
        x_hat = self.decode(z, text_tokens, labels)

        return x_hat, mu, logvar
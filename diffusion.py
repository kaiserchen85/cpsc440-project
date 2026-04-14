import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler

class Diffusion(nn.Module):
    def __init__(
        self, 
        spec_shape=(1, 80, 200), 
        text_vocab_size=10000, 
        text_embed_dim=128, 
        num_classes=2
    ):
        super().__init__()
        self.spec_shape = spec_shape

        # 1. Embeddings
        self.text_embedding = nn.Embedding(text_vocab_size, text_embed_dim)
        self.label_embedding = nn.Embedding(num_classes, 32)

        # 2. The U-Net Denoiser
        # diffusers expects a cross-attention sequence. 
        total_cond_dim = text_embed_dim + 32
        
        self.unet = UNet2DConditionModel(
            sample_size=spec_shape[1:],  # (80, 200)
            in_channels=spec_shape[0],   # 1
            out_channels=spec_shape[0],  # 1
            cross_attention_dim=total_cond_dim, 
            # A lightweight architecture for audio features
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=(
                "CrossAttnDownBlock2D", 
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "CrossAttnUpBlock2D", 
            ),
        )
        
        # 3. The Noise Scheduler (Handles the math of adding/removing noise)
        self.scheduler = DDPMScheduler(num_train_timesteps=200)

    def get_conditioning(self, text_tokens, labels):
        # Embed and pool text: (B, T) -> (B, 1, 128)
        text_emb = self.text_embedding(text_tokens).mean(dim=1).unsqueeze(1)
        
        # Embed label: (B) -> (B, 1, 32)
        label_emb = self.label_embedding(labels).unsqueeze(1)
        
        # Concat into a single sequence for cross-attention: (B, 1, 160)
        return torch.cat([text_emb, label_emb], dim=-1)

    def forward(self, clean_spectrogram, text_tokens, labels):
        """Training pass: Add noise, then try to predict that exact noise."""
        B = clean_spectrogram.shape[0]
        device = clean_spectrogram.device

        # 1. Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device).long()

        # 2. Add random noise to the clean spectrogram
        noise = torch.randn_like(clean_spectrogram)
        noisy_spectrogram = self.scheduler.add_noise(clean_spectrogram, noise, timesteps)

        # 3. Get text + label conditioning
        encoder_hidden_states = self.get_conditioning(text_tokens, labels)

        # 4. Predict the noise
        noise_pred = self.unet(noisy_spectrogram, timesteps, encoder_hidden_states).sample

        # 5. Return the prediction and the target (the actual noise we added)
        return noise_pred, noise

    @torch.no_grad()
    def sample(self, text_tokens, labels, num_inference_steps=50):
        """Inference pass: Start from pure noise and denoise into audio."""
        B = text_tokens.shape[0]
        device = text_tokens.device

        # Start with pure noise matching the spectrogram shape
        audio_shape = (B, self.spec_shape[0], self.spec_shape[1], self.spec_shape[2])
        spectrogram = torch.randn(audio_shape, device=device)

        # Get conditioning
        encoder_hidden_states = self.get_conditioning(text_tokens, labels)

        # Setup scheduler for faster inference
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.scheduler.timesteps:
            # Predict noise for this step
            noise_pred = self.unet(spectrogram, t, encoder_hidden_states).sample
            
            # Compute the previous, slightly less noisy step
            spectrogram = self.scheduler.step(noise_pred, t, spectrogram).prev_sample

        return spectrogram
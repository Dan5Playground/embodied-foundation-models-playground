import torch
import torch.nn as nn
from torchvision import models

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        ])
        # FiLM (Feature-wise Linear Modulation) for conditioning
        self.cond_encoder = nn.Linear(cond_dim, out_channels * 2)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish()

    def forward(self, x, cond):
        # x: [B, C, T], cond: [B, cond_dim]
        out = self.blocks[0](x)
        
        # Inject condition (Vision + Time)
        embed = self.cond_encoder(cond).unsqueeze(-1) # [B, 2*C, 1]
        scale, bias = torch.chunk(embed, 2, dim=1)
        out = self.norm(out) * (1 + scale) + bias
        out = self.act(out)
        
        out = self.blocks[1](out)
        return out + x if x.shape == out.shape else out

class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=2, obs_dim=512, horizon=16):
        super().__init__()
        # 1. Vision Encoder
        self.vision_encoder = models.resnet18(weights='DEFAULT')
        self.vision_encoder.fc = nn.Identity()
        
        # 2. Time Embedding
        self.time_esb = nn.Sequential(
            nn.Linear(1, 128),
            nn.Mish(),
            nn.Linear(128, 128)
        )
        
        # 3. U-Net Blocks (Simplified 1D)
        cond_dim = 512 + 128 # Vision + Time
        self.up_conv = nn.Sequential(
            nn.Conv1d(action_dim, 256, kernel_size=5, padding=2),
            nn.Mish()
        )
        self.mid_block = ConditionalResidualBlock1D(256, 256, cond_dim)
        self.final_conv = nn.Conv1d(256, action_dim, kernel_size=1)

    def forward(self, noisy_actions, timestep, obs_features):
        
        if noisy_actions.ndim == 2:
            noisy_actions = noisy_actions.unsqueeze(0)

        # noisy_actions: [B, 16, 2] -> [B, 2, 16]
        x = noisy_actions.moveaxis(1, 2)
        
        # Embed time
        t_emb = self.time_esb(timestep.float().unsqueeze(-1)) # [B, 128]
        
        # Combine Vision + Time
        obs_cond = torch.cat([obs_features, t_emb], dim=-1) # [B, 640]
        
        # Pass through denoising layers
        x = self.up_conv(x)
        x = self.mid_block(x, obs_cond)
        x = self.final_conv(x)
        
        return x.moveaxis(1, 2) # [B, 16, 2]
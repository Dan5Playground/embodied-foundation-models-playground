import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import clip
import random
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.agents.diffusion_policy_vla import DiffusionPolicy_v2
import os
from tqdm import tqdm

# 1. Linguistic Diversity: Prompt Variants
PROMPT_VARIANTS = [
    "push the T-block to the goal",
    "slide the T-shaped block into the target",
    "move the blue object to the marked area",
    "position the T-block on the target spot",
    "align the T-shaped piece with the goal",
    "maneuver the block into the target zone",
    "nudge the T-shape toward the target"
]

def train_vla():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)
    horizon = 16
    
    # 2. Dataset Setup
    action_deltas = [i/10.0 for i in range(horizon)]
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps={"action": action_deltas, "observation.image": [0.0]})
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. Model & Scheduler Init
    model = DiffusionPolicy_v2(horizon=horizon).to(device)
    model.clip_model.to(device) # Move CLIP weights to GPU/MPS
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    
    # 4. Normalization Stats
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)

    img_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    

    print(f"🚀 Training VLA Policy with Prompt Augmentation...")
    model.train()

    for epoch in range(50):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            obs_img = batch['observation.image'].to(device)
            # add image normalization
            obs_img = img_normalize(obs_img)
            action = (batch['action'].to(device) - action_mean) / action_std
            
            # --- TASK 1: Prompt Augmentation & CFG ---
            # Randomly select a variant
            current_prompt = random.choice(PROMPT_VARIANTS)
            
            # 15% chance to drop prompt for Classifier-Free Guidance training
            if random.random() < 0.15:
                current_prompt = "" 
            
            text_tokens = clip.tokenize([current_prompt]).to(device)
            # ---------------------------------------------------------

            with torch.no_grad():
                obs_features = model.vision_encoder(obs_img).flatten(1)
                text_features = model.clip_model.encode_text(text_tokens).float()
                # Expand text features to match batch size
                text_features = text_features.expand(obs_img.shape[0], -1)
            
            # Standard Diffusion Training Logic
            noise = torch.randn_like(action)
            timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            
            # Forward pass with Language
            noise_pred = model(noisy_actions, timesteps, obs_features, text_features)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"outputs/vla_policy_vla_v2_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_vla()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.agents.diffusion_policy import DiffusionPolicy
import os
from tqdm import tqdm

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)
    horizon = 16
    
    # 1. Dataset with 16-step action window
    action_deltas = [i/10.0 for i in range(horizon)]
    dataset = LeRobotDataset(
        "lerobot/pusht", 
        delta_timestamps={"action": action_deltas, "observation.image": [0.0]}
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Setup stats for Normalization
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)
    
    # 3. Model & Scheduler
    model = DiffusionPolicy(horizon=horizon).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        prediction_type='epsilon'
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, 
                                 num_warmup_steps=500, 
                                 num_training_steps=len(dataloader) * 50)

    model.train()
    for epoch in range(50):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            obs_img = batch['observation.image'].to(device) # Normalized [0,1] by LeRobot
            action = (batch['action'].to(device) - action_mean) / action_std
            
            with torch.no_grad():
                obs_features = model.vision_encoder(obs_img)
                if obs_features.ndim == 4: obs_features = obs_features.flatten(1)
            
            noise = torch.randn_like(action)
            timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            
            noise_pred = model(noisy_actions, timesteps, obs_features)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"outputs/diffusion_policy_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
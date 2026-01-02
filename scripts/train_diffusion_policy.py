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
    # 1. Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)
    horizon = 16
    batch_size = 64
    num_epochs = 50
    
    # 2. Dataset & Action Windowing
    # We want 0.0s (now) plus the next 15 frames (at 10Hz)
    action_deltas = [i/10.0 for i in range(horizon)]
    dataset = LeRobotDataset(
        "lerobot/pusht", 
        delta_timestamps={
            "action": action_deltas,
            "observation.image": [0.0] 
        }
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 3. Model & Noise Scheduler
    model = DiffusionPolicy(horizon=horizon).to(device)
    
    # Standard DDPM Scheduler for Robotics
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        prediction_type='epsilon'
    )
    
    # 4. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    
    # Cosine annealing helps prevent "bouncing" around the local minima at the end
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # normalize inputs
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)
    

    print(f"🚀 Training Diffusion Policy on {device}...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Prepare data
            obs_img = batch['observation.image'].to(device)
            action = batch['action'].to(device) # Shape: [B, 16, 2]

            # 2. NEW: Normalize actions BEFORE adding noise
            # This brings the 0-512 range down to roughly -1 to 1
            action = (action - action_mean) / action_std

            
            # 5. The Training Logic
            # a) Encode vision (frozen ResNet for Day 5 speed)
            with torch.no_grad():
                obs_features = model.vision_encoder(obs_img)
            
            # b) Sample noise that matches the action chunk shape
            noise = torch.randn_like(action) 
            bsz = action.shape[0]
            
            # c) Sample random diffusion timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=device
            ).long()
            
            # d) Forward Diffusion: Corrupt clean actions with noise
            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            
            # e) Predict the noise component
            noise_pred = model(noisy_actions, timesteps, obs_features)
            
            # f) Loss Calculation (MSE between actual noise and predicted noise)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            # 6. Backward Pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"🏁 Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            save_path = f"outputs/diffusion_policy_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()
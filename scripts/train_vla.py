import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.agents.diffusion_policy_vla import DiffusionPolicy
from tqdm import tqdm

def train_vla():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    horizon = 16
    
    # 1. Dataset
    action_deltas = [i/10.0 for i in range(horizon)]
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps={"action": action_deltas, "observation.image": [0.0]})
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Stats & Model
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)
    
    model = DiffusionPolicy(horizon=horizon).to(device)
    model.clip_model.to(device) # Move CLIP to GPU/MPS
    
    # 3. Language Prompt Tokenization
    prompt = "push the T-block to the goal"
    text_tokens = clip.tokenize([prompt]).to(device)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("🚀 Training VLA Agent...")
    for epoch in range(50):
        for batch in tqdm(dataloader):
            obs_img = batch['observation.image'].to(device)
            action = (batch['action'].to(device) - action_mean) / action_std
            
            with torch.no_grad():
                obs_features = model.vision_encoder(obs_img).flatten(1)
                text_features = model.clip_model.encode_text(text_tokens).float()
                # Expand text features to match batch size
                text_features = text_features.expand(obs_img.shape[0], -1)
            
            noise = torch.randn_like(action)
            timesteps = torch.randint(0, 100, (action.shape[0],), device=device).long()
            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            
            # Forward pass with Language
            noise_pred = model(noisy_actions, timesteps, obs_features, text_features)
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), f"outputs/vla_policy_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_vla()
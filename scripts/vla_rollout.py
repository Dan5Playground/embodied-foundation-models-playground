import torch
import numpy as np
import gymnasium as gym
import gym_pusht
import torchvision.transforms as T
import clip
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.agents.diffusion_policy_vla import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gymnasium.wrappers import RecordVideo
import os

def main():
    # 1. Setup Device and Paths
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs("outputs/videos", exist_ok=True)

    # 2. Load Normalization Stats
    # We must use the exact same stats used during training
    dataset = LeRobotDataset("lerobot/pusht")
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)

    # 3. Initialize Model and Load VLA Weights
    model = DiffusionPolicy(horizon=16).to(device)
    # Update this path to your best performing checkpoint
    checkpoint_path = "outputs/vla_policy_epoch_27.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print("⚠️ Checkpoint not found! Ensure training finished.")
    
    model.eval()
    model.clip_model.to(device)

    # 4. Noise Scheduler Settings
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, 
        beta_schedule='squaredcos_cap_v2', 
        prediction_type='epsilon'
    )

    # 5. Environment and Language Prompt
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    env = RecordVideo(env, "outputs/videos", name_prefix="vla_eval")
    
    prompt = "push the T-block to the goal"
    text_tokens = clip.tokenize([prompt]).to(device)

    # Image Transform: Standardized [0, 1] scaling to match training
    img_transform = T.Compose([
        T.ToPILImage(), 
        T.Resize((96, 96)), 
        T.ToTensor()
    ])

    obs, info = env.reset()
    action_history = torch.zeros((16, 16, 2)).to(device)
    
    print(f"🎬 Starting VLA Rollout with prompt: '{prompt}'")

    for t in range(300):
        # A. Process Observation (Vision + Language)
        img_raw = obs['pixels'] if isinstance(obs, dict) else obs
        img_tensor = img_transform(img_raw).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Extract Vision Features
            obs_features = model.vision_encoder(img_tensor).flatten(1)
            # Extract Language Features via CLIP
            text_features = model.clip_model.encode_text(text_tokens).float()
            text_features = text_features.expand(obs_features.shape[0], -1)

        # B. THE DENOISING LOOP (The "Generative" Step)
        # Start with pure Gaussian noise
        noisy_action = torch.randn((1, 16, 2), device=device)
        noise_scheduler.set_timesteps(15) # 15 steps for a balance of speed/precision

        for k in noise_scheduler.timesteps:
            tk = torch.tensor([k], device=device, dtype=torch.long)
            with torch.no_grad():
                # Predict noise based on Image + Text + Time
                noise_pred = model(noisy_action, tk, obs_features, text_features)
                
                # Reverse Diffusion step
                noisy_action = noise_scheduler.step(
                    model_output=noise_pred, 
                    timestep=k, 
                    sample=noisy_action
                ).prev_sample

        # C. Unnormalize Action Plan
        action_plan = (noisy_action.squeeze(0) * action_std) + action_mean

        # D. Temporal Aggregation (EMA Smoothing)
        action_history = torch.roll(action_history, shifts=-1, dims=0)
        action_history[-1] = action_plan
        
        final_action = torch.zeros(2).to(device)
        total_weight = 0
        for j in range(16):
            # Give more weight to newer predictions for the same timestep
            weight = np.exp(-0.25 * (15 - j))
            final_action += action_history[j, 15 - j] * weight
            total_weight += weight
        
        # E. Step Environment
        # Divide by total_weight to normalize the weighted sum
        obs, reward, terminated, truncated, info = env.step((final_action / total_weight).cpu().numpy())
        
        if terminated or truncated:
            break

    env.close()
    print("🏁 VLA Evaluation Complete. Check outputs/videos for the result.")

if __name__ == "__main__":
    main()
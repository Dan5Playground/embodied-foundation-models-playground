import torch
import numpy as np
import gymnasium as gym
import gym_pusht
import torchvision.transforms as T
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.agents.diffusion_policy import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gymnasium.wrappers import RecordVideo
import os

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs("outputs/videos", exist_ok=True)

    # 1. Setup Data Stats (For Unnormalization)
    dataset = LeRobotDataset("lerobot/pusht")
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)

    # 2. Load Model & Scheduler
    # Make sure you point to your latest checkpoint!
    model = DiffusionPolicy(horizon=16).to(device)
    model.load_state_dict(torch.load("outputs/diffusion_policy_epoch_30.pth", map_location=device))
    model.eval()

    # We use the same scheduler as training, but we'll use fewer steps for speed
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        prediction_type='epsilon'
    )

    # 3. Env Setup
    env = gym.make("gym_pusht/PushT-v0", 
                   render_mode="rgb_array", 
                   obs_type="pixels")
    env = RecordVideo(env, "outputs/videos", name_prefix="diffusion_eval")
    
    img_transform = T.Compose([
        T.ToPILImage(), T.Resize((96, 96)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    obs, info = env.reset()
    print("🎬 Starting Diffusion Rollout...")

    # EMA Buffer for temporal smoothing (from Day 4)
    action_history = torch.zeros((16, 16, 2)).to(device)

    for t in range(300):
        # A. Process Observation
        img_tensor = img_transform(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            obs_features = model.vision_encoder(img_tensor)
            if obs_features.ndim == 4: obs_features = obs_features.flatten(1)

        # B. THE DENOISING LOOP
        # 1. Start with pure noise [Batch, Horizon, Action_Dim]
        noisy_action = torch.randn((1, 16, 2), device=device)
        
        # 2. Set inference steps (10 is a good balance of speed vs quality)
        noise_scheduler.set_timesteps(10)

        for k in noise_scheduler.timesteps:
            with torch.no_grad():
                # Predict noise added to the current 'sample'
                timestep = torch.tensor([k], device=device, dtype=torch.long)
                noise_pred = model(noisy_action, timestep, obs_features)
                #noise_pred = model(noisy_action, k, obs_features)
                
                # Compute the 'previous' less-noisy sample
                noisy_action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action
                ).prev_sample

        # C. Unnormalize the final "Clean" result
        # noisy_action is now our 'action_plan' in normalized space
        action_plan = (noisy_action.squeeze(0) * action_std) + action_mean

        # D. Temporal Aggregation (EMA Smoothing)
        action_history = torch.roll(action_history, shifts=-1, dims=0)
        action_history[-1] = action_plan
        
        # Weighted average of overlapping plans
        final_action = torch.zeros(2).to(device)
        total_weight = 0
        for j in range(16):
            time_idx = 15 - j
            weight = np.exp(-0.25 * time_idx)
            final_action += action_history[j, time_idx] * weight
            total_weight += weight
        
        # E. Step Environment
        obs, reward, terminated, truncated, info = env.step(final_action.cpu().numpy())
        
        if terminated or truncated:
            break

    env.close()
    print("🏁 Rollout complete. Video saved to outputs/videos")

if __name__ == "__main__":
    main()
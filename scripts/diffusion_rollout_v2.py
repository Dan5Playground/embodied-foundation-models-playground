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

    dataset = LeRobotDataset("lerobot/pusht")
    action_mean = torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)

    model = DiffusionPolicy(horizon=16).to(device)
    model.load_state_dict(torch.load("outputs/diffusion_policy_epoch_50.pth", map_location=device))
    model.eval()

    noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')

    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    env = RecordVideo(env, "outputs/videos", name_prefix="diffusion_eval")
    
    # CRITICAL: Scaling to [0,1] without extra ImageNet Normalization to match Training
    img_transform = T.Compose([T.ToPILImage(), T.Resize((96, 96)), T.ToTensor()])

    obs, info = env.reset()
    action_history = torch.zeros((16, 16, 2)).to(device)

    for t in range(300):
        img_raw = obs['pixels'] if isinstance(obs, dict) else obs
        img_tensor = img_transform(img_raw).unsqueeze(0).to(device)
        
        with torch.no_grad():
            obs_features = model.vision_encoder(img_tensor)
            if obs_features.ndim == 4: obs_features = obs_features.flatten(1)

        # Iterative Denoising Loop
        noisy_action = torch.randn((1, 16, 2), device=device)
        noise_scheduler.set_timesteps(15) # 15 steps for better precision

        for k in noise_scheduler.timesteps:
            tk = torch.tensor([k], device=device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = model(noisy_action, tk, obs_features)
                noisy_action = noise_scheduler.step(noise_pred, k, noisy_action).prev_sample

        # Unnormalize and Smooth
        action_plan = (noisy_action.squeeze(0) * action_std) + action_mean
        action_history = torch.roll(action_history, shifts=-1, dims=0)
        action_history[-1] = action_plan
        
        # Temporal Aggregation
        final_action = torch.zeros(2).to(device)
        total_weight = 0
        for j in range(16):
            weight = np.exp(-0.25 * (15 - j))
            final_action += action_history[j, 15 - j] * weight
            total_weight += weight
        
        obs, reward, terminated, truncated, info = env.step((final_action/total_weight).cpu().numpy())
        if terminated or truncated: break

    env.close()

if __name__ == "__main__":
    main()
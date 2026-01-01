import torch
import numpy as np
import gymnasium as gym
import gym_pusht
import torchvision.transforms as T
from src.agents.bc_mlp import MLPPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gymnasium.wrappers import RecordVideo
import os

def run_smooth_rollout():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir = "outputs/videos_day4"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Dataset Stats for Unnormalization
    print("📊 Loading metadata stats...")
    dataset = LeRobotDataset("lerobot/pusht")
    stats = dataset.meta.stats['action']
    action_mean = torch.from_numpy(stats['mean']).to(torch.float32).to(device)
    action_std = torch.from_numpy(stats['std']).to(torch.float32).to(device)

    # 2. Load Trained Model
    model = MLPPolicy().to(device)
    model.load_state_dict(torch.load("outputs/bc_mlp_baseline.pth", weights_only=True))
    model.eval()

    # 3. Setup Environment
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    env = RecordVideo(env, video_folder=output_dir, episode_trigger=lambda x: True)
    obs, info = env.reset()

    img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Temporal Aggregation Setup
    chunk_size = 16
    # Buffer to store overlapping plans: [chunk_size, chunk_size, action_dim]
    action_history = torch.zeros((chunk_size, chunk_size, 2)).to(device)
    k = 0.25 # Exponential weight constant (Standard for PushT)

    print(f"🎬 Starting Smooth Rollout (k={k})...")
    
    for t in range(300):
        # Prepare observation
        img_raw = obs#['image']
        img_tensor = img_transform(img_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            # Predict action chunk [1, 16, 2]
            pred_normalized = model(img_tensor)
            # Unnormalize
            new_plan = (pred_normalized.squeeze(0) * action_std) + action_mean

        # --- TEMPORAL AGGREGATION LOGIC ---
        # Shift history: remove oldest plan, add newest plan to the end
        action_history = torch.roll(action_history, shifts=-1, dims=0)
        action_history[-1] = new_plan
        
        # Calculate weighted average for the current timestep
        aggregated_action = torch.zeros(2).to(device)
        total_weight = 0
        
        for i in range(chunk_size):
            # i=0 is the plan from 15 steps ago. i=15 is the plan created 'now'.
            # If a plan was created 'm' steps ago, we want its 'm-th' prediction.
            time_since_plan_start = chunk_size - 1 - i 
            
            # Exponential weighting: newer plans get more weight
            weight = np.exp(-k * time_since_plan_start)
            
            # Extract the coordinate this specific plan intended for 'now'
            action_from_plan = action_history[i, time_since_plan_start]
            
            aggregated_action += action_from_plan * weight
            total_weight += weight
            
        final_action = (aggregated_action / total_weight).cpu().numpy()
        # ----------------------------------
        #print(f"🤖 Robot Target: X={final_action[0]:.2f}, Y={final_action[1]:.2f}")

        obs, reward, terminated, truncated, info = env.step(final_action)
        
        if terminated or truncated:
            break

    env.close()
    print(f"🏁 Smooth rollout complete. Video saved to {output_dir}")

if __name__ == "__main__":
    run_smooth_rollout()
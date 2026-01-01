"""
Simple rollout evaluator for the BC MLP policy on PushT environment.
"""
import torch
import numpy as np
import gymnasium as gym  # Use gymnasium instead of gym if possible
import gym_pusht        # This MUST be imported to register the env
import torchvision.transforms as T
from src.agents.bc_mlp import MLPPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gymnasium.wrappers import RecordVideo

def run_rollout():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Dataset Stats (Needed for unnormalization)
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
    env = gym.make(
                "gym_pusht/PushT-v0", 
                render_mode="rgb_array", 
                obs_type="pixels"  
            )
    env = RecordVideo(env, video_folder="outputs/videos", episode_trigger=lambda x: True)
    obs, info = env.reset()

    # Match the normalization used in training
    img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("🎬 Starting Simulation. Watch the PyGame window!")
    
    for _ in range(300): # Run for 300 steps
        if isinstance(obs, dict):
            print("Observation is a dict.")
            img = obs['image']  # Use this if it's a dictionary
        else:
            img = obs           # Use this if gym-pusht is returning the array directly
        # Prepare image: [H, W, C] -> [1, 3, 96, 96]
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Predict action chunk [1, 16, 2]
            pred_normalized = model(img_tensor)
            
            # Unnormalize: Action = (Pred * Std) + Mean
            pred_action = (pred_normalized * action_std) + action_mean
            
            # Execute only the FIRST action of the chunk (Receding Horizon)
            action = pred_action[0, 0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

    env.close()
    print("🏁 Rollout complete.")

if __name__ == "__main__":
    run_rollout()
import torch
import gymnasium as gym
import gym_pusht
import torchvision.transforms as T
from src.agents.bc_mlp import MLPPolicy
from src.utils.evaluation import evaluate_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gymnasium.wrappers import RecordVideo
import os

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    video_dir = "outputs/videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Load model and stats
    dataset = LeRobotDataset("lerobot/pusht")
    
    # Applying the remembered casting order: .to(float32).to(device)
    action_stats = {
        'mean': torch.from_numpy(dataset.meta.stats['action']['mean']).to(torch.float32).to(device),
        'std': torch.from_numpy(dataset.meta.stats['action']['std']).to(torch.float32).to(device)
    }
    
    model = MLPPolicy().to(device)
    model.load_state_dict(torch.load("outputs/bc_mlp_baseline.pth", weights_only=True))
    model.eval()

    # Initialize environment with video recording
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix="compare")
    
    # Updated transform to handle 96x96x3 numpy arrays
    img_transform = T.Compose([
        T.ToPILImage(), 
        T.Resize((96, 96)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- 🏁 Starting Comparison Test ---")
    
    # Test 1: Basic
    print("\nRunning Basic Controller...")
    sr_basic, rew_basic = evaluate_policy(
        env, model, img_transform, action_stats, device, 
        num_episodes=5, controller_type="basic"
    )
    
    # Test 2: EMA (Smooth)
    print("\nRunning EMA Controller...")
    sr_ema, rew_ema = evaluate_policy(
        env, model, img_transform, action_stats, device, 
        num_episodes=5, controller_type="ema", k=0.25
    )

    print("\n" + "="*30)
    print(f"FINAL RESULTS (5 Episodes each):")
    print(f"BASIC: Success Rate: {sr_basic}% | Avg Max Reward: {rew_basic:.3f}")
    print(f"EMA:   Success Rate: {sr_ema}% | Avg Max Reward: {rew_ema:.3f}")
    print("="*30)
    
    env.close() # Vital for flushing video buffers

if __name__ == "__main__":
    main()
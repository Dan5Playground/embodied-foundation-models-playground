import torch
import numpy as np
import gymnasium as gym
import gym_pusht
from src.agents.mlp_policy import MLPPolicy
from src.agents.diffusion_policy import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def evaluate_policy(env, policy, policy_type="diffusion", num_episodes=5):
    total_rewards = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        # Diffusion requires a noisy starting action to denoise
        # MLP just takes the observation
        while not done:
            with torch.no_grad():
                if policy_type == "diffusion":
                    # Run iterative denoising loop
                    action = policy.generate_action(obs) 
                else:
                    # Single forward pass
                    action = policy(obs) 
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def main():
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
    
    # Load standardized stats for normalization
    dataset = LeRobotDataset("lerobot/pusht")
    stats = dataset.meta.stats
    
    # Initialize Policies
    policies = {
        "MLP": MLPPolicy(hidden_dim=256),
        "Diffusion": DiffusionPolicy(horizon=16),
        "VLA": DiffusionPolicy(use_language=True) # VLA is Diffusion + CLIP
    }

    results = {}
    for name, agent in policies.items():
        # Load your saved .pth checkpoints here
        # agent.load_state_dict(torch.load(f"outputs/{name}_best.pth"))
        avg_score = evaluate_policy(env, agent, policy_type=name.lower())
        results[name] = avg_score
        print(f"📊 {name} Average Reward: {avg_score:.2f}")

if __name__ == "__main__":
    main()
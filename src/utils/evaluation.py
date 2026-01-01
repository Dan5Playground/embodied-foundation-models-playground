import torch
import numpy as np

def evaluate_policy(env, model, img_transform, action_stats, device, 
                    num_episodes=10, controller_type="basic", k=0.25):
    """
    Modular evaluator that can handle both 'basic' and 'ema' (Smooth) control.
    """
    successes = 0
    results = []
    chunk_size = 16
    
    action_mean = action_stats['mean'].to(torch.float32).to(device)
    action_std = action_stats['std'].to(torch.float32).to(device)

    for i in range(num_episodes):
        obs, info = env.reset()
        max_reward = 0
        # Buffer for EMA
        action_history = torch.zeros((chunk_size, chunk_size, 2)).to(device)
        
        for t in range(300):
            img_tensor = img_transform(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(img_tensor)
                new_plan = (pred_norm.squeeze(0) * action_std) + action_mean

            # --- SELECTION LOGIC ---
            if controller_type == "basic":
                # Just take the first action of the newest plan
                action = new_plan[0].cpu().numpy()
            
            elif controller_type == "ema":
                # Apply the Temporal Aggregation logic
                action_history = torch.roll(action_history, shifts=-1, dims=0)
                action_history[-1] = new_plan
                
                aggregated_action = torch.zeros(2).to(device)
                total_weight = 0
                for j in range(chunk_size):
                    time_idx = chunk_size - 1 - j
                    weight = np.exp(-k * time_idx)
                    aggregated_action += action_history[j, time_idx] * weight
                    total_weight += weight
                action = (aggregated_action / total_weight).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            max_reward = max(max_reward, reward)
            
            if terminated or truncated:
                break
        
        is_success = max_reward > 0.9
        if is_success: successes += 1
        results.append(max_reward)
        print(f"[{controller_type.upper()}] Episode {i}: {'✅' if is_success else '❌'} Max Reward: {max_reward:.2f}")

    avg_reward = sum(results) / num_episodes
    success_rate = (successes / num_episodes) * 100
    return success_rate, avg_reward
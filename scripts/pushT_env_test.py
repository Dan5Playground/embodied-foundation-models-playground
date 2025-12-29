import gymnasium as gym
import gym_pusht
import imageio
import numpy as np

print("🤖 Initializing Push-T Environment...")
# We use 'pixels' so we can train Vision models later
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")

obs, info = env.reset()
frames = []

print("🕹️ Running 50 random steps...")
for _ in range(50):
    # Action is [x, y] velocity
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Capture the frame
    frame = env.render()
    frames.append(frame)
    
    if terminated or truncated:
        break

env.close()

# Save the video so you can add it to your GitHub Day 1
imageio.mimsave("day1_test.mp4", frames, fps=10)
print("✅ Success! Video saved as 'day1_test.mp4'. Environment is ready.")

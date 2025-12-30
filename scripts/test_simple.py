# try:
#     from lerobot.datasets.lerobot_dataset import LeRobotDataset
#     print("✅ LeRobot found!")
#     dataset = LeRobotDataset("lerobot/pusht")
#     print(f"✅ Dataset loaded: {len(dataset)} frames")
# except ImportError as e:
#     print(f"❌ Error: {e}")

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def test_dataset():
    print("🎬 Loading LeRobot Push-T dataset...")
    dataset = LeRobotDataset("lerobot/pusht")
    
    # 1. Check basic info
    print(f"📊 Total Frames: {len(dataset)}")
    print(f"🎞️ Total Episodes: {dataset.num_episodes}")

    # 2. Test Frame Extraction (This tests the TorchCodec/FFmpeg link again)
    print("🔄 Testing frame extraction (Frame 0)...")
    try:
        sample = dataset[0]
        action = sample['action']
        observation = sample['observation.image'] # RGB Camera data
        
        print(f"✅ Action Shape: {action.shape} (Expected: [2])")
        print(f"✅ Image Shape:  {observation.shape} (Expected: [3, 96, 96])")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return

    # 3. Test Episode extraction
    print("🔄 Testing episode extraction...")
    # 3. Test Episode extraction
    print("🔄 Testing episode extraction...")
    
    episode_to_find = 0
    
    # This searches the metadata for frames belonging to Episode 0
    # We find where 'episode_index' == 0
    indices = [i for i, ep in enumerate(dataset.hf_dataset['episode_index']) if ep == episode_to_find]
    
    if not indices:
        print("❌ Could not find episode indices.")
        return

    actions = []
    # Grab the first 20 frames of this episode
    for i in indices[:20]:
        actions.append(dataset[i]['action'])
    
    actions_tensor = torch.stack(actions)
    print(f"✅ Episode Slice Shape: {actions_tensor.shape} (Expected: [20, 2])")
    print("\n🚀 DATA PIPELINE VERIFIED!")

if __name__ == "__main__":
    test_dataset()
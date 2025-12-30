"""
pip install -e .
python scripts/data_inspection.py
"""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.utils.chunking import compute_action_chunks

def main():
    # 1. Load the dataset (This downloads it if missing)
    print("📥 Loading LeRobot Push-T dataset...")
    dataset = LeRobotDataset("lerobot/pusht")
    
    # 2. Extract actions from the first episode
    # In LeRobot datasets, 'action' is stored per frame.
    # We'll grab the first 100 frames to test our chunker.
    print("🔄 Extracting first 100 frames...")
    actions_list = []
    for i in range(100):
        # dataset[i] is a dict; we grab the 'action' tensor
        frame_action = dataset[i]['action'] 
        actions_list.append(frame_action)
    
    # 3. Stack into a single [100, 2] tensor
    actions_tensor = torch.stack(actions_list)
    # all_actions = dataset.hf_dataset.select(range(100))#['action']
    # #print(all_actions.shape, all_actions.type)
    # #actions_tensor = torch.tensor(all_actions_list, dtype=torch.float32)
    # all_actions = all_actions['action']
    # actions_tensor = torch.tensor(all_actions, dtype=torch.float32)
    
    print(f"✅ Extracted raw actions shape: {actions_tensor.shape}") # [100, 2]

    # 3. Apply your Chunking Logic
    chunk_size = 16
    chunks = compute_action_chunks(actions_tensor, chunk_size)
    
    print(f"🧱 Chunked actions shape: {chunks.shape}") 
    print(f"💡 Explanation: We now have {chunks.shape[0]} windows, "
          f"each containing {chunk_size} future steps.")

    # 4. Look at a single chunk
    print("\n🧐 Inspection of Chunk #0 (First 3 steps):")
    print(chunks[0][:3])

if __name__ == "__main__":
    main()
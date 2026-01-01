import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PushTDataset(Dataset):
    def __init__(self, lerobot_dataset, chunk_size=16):
        self.dataset = lerobot_dataset
        self.chunk_size = chunk_size
        # Action normalization stats
        self.action_stats = self.dataset.meta.stats['action']
        self.action_mean = torch.from_numpy(self.action_stats['mean']).to(torch.float32)
        self.action_std = torch.from_numpy(self.action_stats['std']).to(torch.float32)
        
        # ResNet18 standard normalization
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # We stop early so we don't grab a chunk that goes past the end
        return len(self.dataset) - self.chunk_size

    # def __getitem__(self, idx):
    #     # 1. Get current image (Observation)
    #     image = self.dataset[idx]['observation.image']
    #     image = self.transform(image)
        
    #     # 2. Get next 16 actions (Action Chunk)
    #     actions = []
    #     for i in range(idx, idx + self.chunk_size):
    #         actions.append(self.dataset[i]['action'])
        
    #     action_chunk = torch.stack(actions)
        
    #     return image, action_chunk
    # vectorized version
    def __getitem__(self, idx):
        # 1. Get current image 
        image = self.dataset[idx]['observation.image']
        image = self.transform(image)
        
        # 2. VECTORIZED ACTION SLICE (Fast!)
        # Reach into the hf_dataset directly to get a block of 16 rows
        action_chunk = self.dataset.hf_dataset[idx : idx + self.chunk_size]['action']
        
        action_chunk = torch.stack(action_chunk).to(torch.float32)

        # 3. APPLY NORMALIZATION HERE
        # (Value - Mean) / Std
        normalized_actions = (action_chunk - self.action_mean) / self.action_std
        
        return image, normalized_actions
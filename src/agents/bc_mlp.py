import torch
import torch.nn as nn
from torchvision import models

class MLPPolicy(nn.Module):
    def __init__(self, action_dim=2, chunk_size=16, hidden_dim=256):
        super().__init__()
        
        # 1. Vision Encoder: Pre-trained ResNet18
        # We strip the final classification layer to get a feature vector
        resnet = models.resnet18(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18 output is 512-dimensional
        self.feature_dim = 512
        
        # 2. Policy Head: MLP
        # Input: Visual Features | Output: Chunk_Size * Action_Dim
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim)
        )
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def forward(self, image):
        """
        Args:
            image: Batch of RGB images (B, 3, 96, 96)
        Returns:
            action_chunks: (B, chunk_size, action_dim)
        """
        # Extract features: (B, 512, 1, 1) -> (B, 512)
        features = self.visual_encoder(image).flatten(1)
        
        # Predict flattened actions: (B, chunk_size * action_dim)
        out = self.mlp(features)
        
        # Reshape to chunked format: (B, 16, 2)
        return out.view(-1, self.chunk_size, self.action_dim)

    def unnormalize_action(self, action, stats):
        """
        Converts model output back to environment coordinates.
        action: (..., 2)
        stats: dictionary with 'mean' and 'std'
        """
        return (action * stats['std'].to(action.device)) + stats['mean'].to(action.device)

if __name__ == "__main__":
    # Quick Shape Test
    model = MLPPolicy()
    test_img = torch.randn(1, 3, 96, 96)
    output = model(test_img)
    print(f"Input Shape: {test_img.shape}")
    print(f"Output Action Chunk Shape: {output.shape}") # Should be (1, 16, 2)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.agents.bc_mlp import MLPPolicy
from src.utils.dataset_utils import PushTDataset

def train():
    # 1. Setup Device (M3 GPU acceleration)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. Load Data
    raw_dataset = LeRobotDataset("lerobot/pusht")
    train_dataset = PushTDataset(raw_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = MLPPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # We want to minimize the distance between pred and expert

    # 4. Simple Training Loop
    model.train()
    for epoch in range(50): # Start small
        epoch_loss = 0
        for images, expert_actions in train_loader:
            images, expert_actions = images.to(device), expert_actions.to(device)

            # Forward pass
            pred_actions = model(images)
            loss = criterion(pred_actions, expert_actions)

            # Backward pass (Optimization)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"📉 Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.6f}")

    # 5. Save the weights
    torch.save(model.state_dict(), "outputs/bc_mlp_baseline.pth")
    print("✅ Training complete. Weights saved to outputs/")

if __name__ == "__main__":
    train()
import torch

def compute_action_chunks(actions, chunk_size):
    """
    Groups a sequence of actions into overlapping chunks.
    
    Args:
        actions (torch.Tensor): Shape (N, action_dim) 
        chunk_size (int): How many future steps to predict (e.g., 16)
        
    Returns:
        chunks (torch.Tensor): Shape (N - chunk_size + 1, chunk_size, action_dim)
    """
    if len(actions) < chunk_size:
        return actions.unsqueeze(0)
    
    # unfold(dimension, size, step)
    # This creates a sliding window view of the tensor
    chunks = actions.unfold(0, chunk_size, 1)
    
    # PyTorch unfold puts the window size at the end, so we permute it back
    # Shape becomes: (Num_Chunks, Action_Dim, Chunk_Size) -> (Num_Chunks, Chunk_Size, Action_Dim)
    chunks = chunks.permute(0, 2, 1)
    
    return chunks

# Test the logic
if __name__ == "__main__":
    test_actions = torch.randn(100, 2) # 100 steps of x,y actions
    output = compute_action_chunks(test_actions, 16)
    print(f"Input shape: {test_actions.shape}")
    print(f"Chunked shape: {output.shape}") # Should be (85, 16, 2)
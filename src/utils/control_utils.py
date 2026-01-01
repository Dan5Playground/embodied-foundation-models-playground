import torch
import numpy as np

class TemporalAggregator:
    def __init__(self, chunk_size=16, device="mps"):
        self.chunk_size = chunk_size
        self.device = device
        # This buffer will store overlapping predictions
        self.action_buffer = torch.zeros((chunk_size, chunk_size, 2)).to(device)
        self.step_idx = 0

    def add_prediction(self, action_chunk):
        """
        action_chunk: [1, 16, 2] tensor
        """
        # Shift old predictions (moving them 1 step into the past)
        self.action_buffer = torch.roll(self.action_buffer, shifts=-1, dims=0)
        # Clear the oldest row
        self.action_buffer[-1] = torch.zeros((self.chunk_size, 2)).to(self.device)
        # Add the new 16-step plan
        self.action_buffer[-1] = action_chunk.squeeze(0)

    def get_aggregated_action(self):
        """
        Returns a single [2] action by averaging overlapping plans.
        """
        # We look at the action_buffer and pick the valid overlapping steps
        # For simplicity, we'll start with a linear weighted average
        valid_actions = []
        for i in range(self.chunk_size):
            # The i-th plan's prediction for 'now'
            # (This is a bit tricky to index, we'll refine this in the script)
            pass
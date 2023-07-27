import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_output, positive_output, negative_output):
        distance_positive = torch.norm(anchor_output - positive_output, p=2, dim=1)
        distance_negative = torch.norm(anchor_output - negative_output, p=2, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)

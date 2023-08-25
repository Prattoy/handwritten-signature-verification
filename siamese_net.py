import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, batch_size, num_heads, num_layers, dropout=0.1):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # Define the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Define the architecture of the Siamese network using Transformer
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        # Calculate the number of features after Transformer layers
        self.transformer_features = embedding_dim  # This can be adjusted based on your requirements

        # Fully connected layer for producing the final embeddings
        self.fc1 = nn.Linear(self.transformer_features, embedding_dim)

    def forward_once(self, x):
        # x = x.view(self.batch_size, 64, 256)
        # x = x.transpose(0, 1)  # Transpose dimensions to match Transformer input format
        # x = self.transformer_encoder(x)
        # x = x.transpose(0, 1)  # Transpose dimensions back to batch-first
        # x = x.mean(dim=1)  # Average over sequence length to get a fixed-size representation
        # x = F.relu(self.fc1(x))

        x = self.cnn(x)
        # print(x.shape)

        x = x.flatten(2).transpose(1, 2)  # Flatten the image features and transpose for Transformer
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over sequence length to get a fixed-size representation
        x = F.relu(self.fc1(x))
        return x

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        return f"SiameseNetwork(embedding_dim={self.embedding_dim}, num_params={self.count_parameters()})"

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        # Define the architecture of the Siamese network using Transformer
        self.embedding = nn.Embedding(256, embedding_dim)  # Embedding layer for input signatures
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Calculate the number of features after Transformer layers
        self.transformer_features = embedding_dim  # This can be adjusted based on your requirements

        # Fully connected layer for producing the final embeddings
        self.fc1 = nn.Linear(self.transformer_features, embedding_dim)

    def forward_once(self, x):
        x = x.long()  # Convert input to LongTensor
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Transpose dimensions to match Transformer input format
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Transpose dimensions back to batch-first
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

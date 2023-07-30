import torch.nn as nn
import torch.nn.functional as f


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        # Define the architecture of the Siamese network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Update input channels to 3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1204224, embedding_dim)  # Adjust the input size for the fully connected layer

    def forward_once(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor to a 1D tensor

        # Print the size of x to get the input_size for fc1
        # print(x.size())

        x = f.relu(self.fc1(x))
        return x

    def forward(self, anchor, positive, negative):
        # print(anchor.shape())
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        # Provide a more descriptive representation when printing the object
        return f"SiameseNetwork(embedding_dim={self.embedding_dim}, num_params={self.count_parameters()})"

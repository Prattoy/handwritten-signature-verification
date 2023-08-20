import torch
import random
from preprocess import preprocess_data
from transformer import extract_embeddings
from siamese_net import SiameseNetwork
from triplet_loss import TripletLoss
import time


# def create_signature_triplets(embeddings_list, margin=1.0):
#     triplets = []
#     labels = []
#
#     for embeddings in embeddings_list:
#         num_embeddings = embeddings.shape[0]
#
#         for i in range(num_embeddings):
#             anchor = embeddings[i]
#
#             # Choose a random positive sample (from the same person)
#             positive_idx = random.choice([idx for idx in range(num_embeddings) if idx != i])
#             positive = embeddings[positive_idx]
#
#             # Choose a random negative sample (from a different person)
#             negative_idx = random.choice([idx for idx in range(num_embeddings) if idx != i])
#             negative = embeddings[negative_idx]
#
#             # Append the triplet and corresponding label
#             triplets.append((anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)))  # Unsqueeze to add batch dimension
#             labels.append(1)
#
#     return triplets, torch.tensor(labels)


def train_siamese_network(anchor_dataloader, positive_dataloader, negative_dataloader, desired_embedding_dim, num_epochs=10):
    # Initialize the Siamese network and TripletLoss
    siamese_net = SiameseNetwork(embedding_dim=desired_embedding_dim, num_heads=4, num_layers=2)  # Use the SiameseNetwork with Transformer
    criterion = TripletLoss(margin=0.5)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0

        siamese_net.train()  # Set the model to training mode
        for batch_idx, (anchors, positives, negatives) in enumerate(
                zip(anchor_dataloader, positive_dataloader, negative_dataloader)):
            optimizer.zero_grad()

            # # Assuming anchors, positives, and negatives are tensors
            # anchor_sequence_length = anchors.size(0)
            # positive_sequence_length = positives.size(0)
            # negative_sequence_length = negatives.size(0)
            #
            # # Compare sequence lengths
            # if anchor_sequence_length == positive_sequence_length == negative_sequence_length:
            #     print("All sequence lengths are the same.")
            # else:
            #     print("Sequence lengths are different.")

            # Reshape the tensors to [batch_size, num_channels * height * width]
            anchors = anchors.view(anchors.size(0), -1)  # The -1 automatically calculates the necessary size
            positives = positives.view(positives.size(0), -1)
            negatives = negatives.view(negatives.size(0), -1)

            # Assuming anchors, positives, and negatives are your input tensors
            print("Anchors shape:", anchors.size())
            print("Positives shape:", positives.size())
            print("Negatives shape:", negatives.size())

            # Forward pass through the Siamese Network
            anchor_output, positive_output, negative_output = siamese_net(anchors, positives, negatives)

            # Compute the triplet loss
            loss = criterion(anchor_output, positive_output, negative_output)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(anchor_dataloader)  # Calculate the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")


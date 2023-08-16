import torch
import random
from preprocess import preprocess_data
from transformer import extract_embeddings
from siamese_net import SiameseNetwork
from triplet_loss import TripletLoss


def create_signature_triplets(embeddings_list, margin=1.0):
    triplets = []
    labels = []

    for embeddings in embeddings_list:
        num_embeddings = embeddings.shape[0]

        for i in range(num_embeddings):
            anchor = embeddings[i]

            # Choose a random positive sample (from the same person)
            positive_idx = random.choice([idx for idx in range(num_embeddings) if idx != i])
            positive = embeddings[positive_idx]

            # Choose a random negative sample (from a different person)
            negative_idx = random.choice([idx for idx in range(num_embeddings) if idx != i])
            negative = embeddings[negative_idx]

            # Append the triplet and corresponding label
            triplets.append((anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)))  # Unsqueeze to add batch dimension
            labels.append(1)

    return triplets, torch.tensor(labels)



def train_siamese_network(signature_dataloader, desired_embedding_dim, num_epochs=10):
    # Initialize the Siamese network and TripletLoss
    siamese_net = SiameseNetwork(embedding_dim=desired_embedding_dim)  # Modify as per your Siamese network architecture
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        siamese_net.train()  # Set the model to training mode
        signature_embeddings_list = []
        for batch_idx, batch_images in enumerate(signature_dataloader):
            signature_embeddings = extract_embeddings(batch_images)
            signature_embeddings_list.append(signature_embeddings)
            break

        # Create triplets and corresponding labels for training using all batches
        triplets, labels = create_signature_triplets(signature_embeddings_list)

        # Convert triplets and labels to tensors
        anchor_tensors, positive_tensors, negative_tensors = zip(*triplets)

        anchors = torch.stack(anchor_tensors)
        print(anchors.shape)
        positives = torch.stack(positive_tensors)
        negatives = torch.stack(negative_tensors)
        labels = labels.clone().detach()
        optimizer.zero_grad()

        # Forward pass through the Siamese Network
        anchor_output, positive_output, negative_output = siamese_net(anchors, positives, negatives)

        # Compute the triplet loss
        loss = criterion(anchor_output, positive_output, negative_output)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        epoch_loss = running_loss / len(signature_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def main():
    data_folder = "/Users/prattoymajumder/Documents/SignaturerDatasets/Dataset_1/signatures/full_forg"
    image_size = (128, 128)
    batch_size = 32
    desired_embedding_dim = 197

    signature_dataloader = preprocess_data(data_folder, image_size=image_size, batch_size=batch_size)

    # Train the Siamese network
    train_siamese_network(signature_dataloader, desired_embedding_dim, num_epochs=10)


if __name__ == "__main__":
    main()

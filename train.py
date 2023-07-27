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
        # print(num_embeddings)

        for i in range(num_embeddings):
            anchor = embeddings[i]

            # Choose a random positive sample (from the same person)
            positive_idx = random.choice([idx for idx in range(num_embeddings) if idx != i])
            positive = embeddings[positive_idx]

            # Choose a random negative sample (from a different person)
            negative_idx = random.choice([idx for idx in range(num_embeddings) if idx != i])
            negative = embeddings[negative_idx]

            # Append the triplet and corresponding label
            triplets.append((anchor, positive, negative))
            labels.append(1)

    return triplets, torch.tensor(labels)


def train_siamese_network(siamese_net, dataloader, criterion, optimizer, num_epochs=10):
    siamese_net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):

            anchor, positive, negative, labels = data

            optimizer.zero_grad()
            anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)

            # Note: You can ignore the 'labels' tensor from the DataLoader as it is not used in training
            # The 'labels' tensor was created while generating triplets but is not needed during training

            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def main():
    data_folder = "/Users/prattoymajumder/Documents/SignaturerDatasets/Dataset_1/signatures/full_forg"
    image_size = (128, 128)
    batch_size = 32

    signature_dataloader = preprocess_data(data_folder, image_size=image_size, batch_size=batch_size)

    # Assuming 'signature_dataloader' contains preprocessed signature images
    signature_embeddings_list = []

    for batch_images in signature_dataloader:
        signature_embeddings = extract_embeddings(batch_images)
        signature_embeddings_list.append(signature_embeddings)

    # Create triplets and corresponding labels for training
    triplets, labels = create_signature_triplets(signature_embeddings_list)

    # Initialize the Siamese network and TripletLoss
    siamese_net = SiameseNetwork(
        embedding_dim=signature_embeddings.size(1))  # Modify as per your Siamese network architecture
    criterion = TripletLoss(margin=1.0)

    # Training loop
    num_epochs = 10
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)

    # Train the Siamese network
    train_siamese_network(siamese_net, triplets, criterion, optimizer, num_epochs)


if __name__ == "__main__":
    main()

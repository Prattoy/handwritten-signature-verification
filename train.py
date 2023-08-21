import torch
import random
from preprocess import preprocess_data
from transformer import extract_embeddings
from siamese_net import SiameseNetwork
from triplet_loss import TripletLoss
import time
import torch.nn.functional as f
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def train_siamese_network(siamese_net, anchor_dataloader, positive_dataloader, negative_dataloader, num_epochs=10):
    # Initialize the Siamese network and TripletLoss
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

            # Reshape the tensors to [batch_size, num_channels * height * width]
            anchors = anchors.view(anchors.size(0), -1)  # The -1 automatically calculates the necessary size
            positives = positives.view(positives.size(0), -1)
            negatives = negatives.view(negatives.size(0), -1)

            # Assuming anchors, positives, and negatives are your input tensors
            # print("Anchors shape:", anchors.size())
            # print("Positives shape:", positives.size())
            # print("Negatives shape:", negatives.size())

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


def test_siamese_network(siamese_net, validation_dataloader):
    siamese_net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        all_true_labels = []
        all_predicted_labels = []

        for batch_idx, (anchors, positives, negatives) in enumerate(validation_dataloader):
            # Reshape the tensors as before
            anchors = anchors.view(anchors.size(0), -1)
            positives = positives.view(positives.size(0), -1)
            negatives = negatives.view(negatives.size(0), -1)

            # Forward pass through the Siamese Network
            anchor_output, positive_output, negative_output = siamese_net(anchors, positives, negatives)

            # Calculate the distances
            distance_positive = f.pairwise_distance(anchor_output, positive_output)
            distance_negative = f.pairwise_distance(anchor_output, negative_output)

            # Predicted labels
            predicted_labels = (distance_positive < distance_negative).cpu().numpy()
            all_predicted_labels.extend(predicted_labels)

            # True labels (1 for positive pairs, 0 for negative pairs)
            true_labels = torch.ones_like(distance_positive).cpu().numpy()
            all_true_labels.extend(true_labels)

        # Calculate precision, recall, f1-score, and accuracy
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='binary')
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2%}")




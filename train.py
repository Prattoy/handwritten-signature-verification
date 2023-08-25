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

            # print(anchors.shape)
            # Reshape the tensors to [batch_size, num_channels * height * width]
            # anchors = anchors.view(anchors.size(0), -1)  # The -1 automatically calculates the necessary size
            # positives = positives.view(positives.size(0), -1)
            # negatives = negatives.view(negatives.size(0), -1)

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
            # print(loss.item())

        # print(len(anchor_dataloader))
        epoch_loss = running_loss / len(anchor_dataloader)  # Calculate the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")


def test_siamese_network(siamese_net, test_dataloader):
    siamese_net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correctly_predicted = 0
        incorrectly_predicted = 0
        all_true_labels = []
        all_predicted_labels = []

        for anchors, positives, negatives in test_dataloader:

            anchors = anchors.unsqueeze(0)  # Add batch dimension
            positives = positives.unsqueeze(0)  # Add batch dimension
            negatives = negatives.unsqueeze(0)  # Add batch dimension
            # print(anchors.shape)

            # anchor_signature = anchor_signature.view(1, -1)  # Reshape the anchor signature
            # test_signature = test_signature.view(1, -1)  # Reshape the test signature

            # Forward pass through the Siamese Network
            random_number = random.randint(0, 1)

            if random_number == 0:
                anchor_output = siamese_net.forward_once(anchors)
                test_output = siamese_net.forward_once(negatives)
                # true_label = False
            else:
                anchor_output = siamese_net.forward_once(anchors)
                test_output = siamese_net.forward_once(positives)
                # true_label = True

            # print(anchor_output)
            # print(test_output)
            # Calculate the distance between anchor and test signatures
            distance = f.pairwise_distance(anchor_output, test_output)

            # If the distance is below a certain threshold, consider the test signature as genuine (positive)
            # Otherwise, consider it as forged (negative)
            threshold = 0.5
            predicted_label = 1 if distance < threshold else 0

            all_true_labels.append(random_number)
            all_predicted_labels.append(predicted_label)

            print("Distance: {:.4f}".format(distance.item()))
            print(f"Predicted Label: {predicted_label}")
            print(f"Actual Label: {random_number}")

            if predicted_label == random_number:
                correctly_predicted += 1
            else:
                incorrectly_predicted += 1

        print(f"Total Correctly Predicted: {correctly_predicted}")
        print(f"Total Incorrectly Predicted: {incorrectly_predicted}")

        # Calculate precision, recall, f1-score, and accuracy
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels,
                                                                   average='binary')
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2%}")



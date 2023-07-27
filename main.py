from preprocess import preprocess_data
from transformer import extract_embeddings
from visualize import visualize_signature_images
import train as t
from siamese_net import SiameseNetwork
from triplet_loss import TripletLoss
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset


def main():
    data_folder = "/Users/prattoymajumder/Documents/SignaturerDatasets/Dataset_1/signatures/full_forg"
    image_size = (256, 256)
    batch_size = 32

    signature_dataloader = preprocess_data(data_folder, image_size=image_size, batch_size=batch_size)

    # num_batches = len(signature_dataloader)
    # print(f"Number of batches in signature_dataloader: {num_batches}")

    # to visualize if the data has been preprocessed or not
    # visualize_signature_images(signature_dataloader)

    # Assuming 'signature_dataloader' contains preprocessed signature images
    signature_embeddings_list = []

    for batch_images in signature_dataloader:
        signature_embeddings = extract_embeddings(batch_images)
        signature_embeddings_list.append(signature_embeddings)

        # Create triplets and corresponding labels for training
        triplets, labels = t.create_signature_triplets(signature_embeddings_list)

        # Convert triplets and labels to tensors
        anchor_tensors, positive_tensors, negative_tensors = zip(*triplets)
        anchor_tensors = torch.stack(anchor_tensors)
        positive_tensors = torch.stack(positive_tensors)
        negative_tensors = torch.stack(negative_tensors)
        labels = labels.clone().detach()

        # Create the training dataset using DataLoader
        train_dataset = TensorDataset(anchor_tensors, positive_tensors, negative_tensors, labels)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the Siamese network and TripletLoss
        siamese_net = SiameseNetwork(
            embedding_dim=signature_embeddings.size(1))  # Modify as per your Siamese network architecture
        criterion = TripletLoss(margin=1.0)

        # to check if the siamese net has been initialized or not
        # print(siamese_net)  # Print the model to see its architecture

        # Training loop
        num_epochs = 10
        optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)
        # print(optimizer)

        # Train the Siamese network
        t.train_siamese_network(siamese_net, dataloader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()

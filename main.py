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
    data_folder_path = "/Users/prattoymajumder/Documents/test_dataset"
    image_size = (128, 128)
    batch_size = 32
    num_epochs = 10
    desired_embedding_dim = 197

    signature_dataloader = preprocess_data(data_folder_path, image_size=image_size, batch_size=batch_size)

    num_batches = len(signature_dataloader)
    print(f"Number of batches in signature_dataloader: {num_batches}")
    # print(signature_dataloader[0].shape())
    # to visualize if the data has been preprocessed or not
    # visualize_signature_images(signature_dataloader)


    # signature_embeddings = extract_embeddings(signature_dataset)
    # signature_embeddings_list.append(signature_embeddings)


    # Train the Siamese network
    t.train_siamese_network(signature_dataloader, desired_embedding_dim, num_epochs)

if __name__ == "__main__":
    main()

from preprocess import preprocess_data
from visualize import visualize_signature_images
import train as t


def main():
    anchor_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/anchor"
    positive_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/positive"
    negative_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/negative"
    image_size = (128, 128)
    batch_size = 1
    num_epochs = 1
    desired_embedding_dim = 128

    # preprocess data
    anchor_dataloader = preprocess_data(anchor_folder_path, image_size=image_size, batch_size=batch_size)
    positive_dataloader = preprocess_data(positive_folder_path, image_size=image_size, batch_size=batch_size)
    negative_dataloader = preprocess_data(negative_folder_path, image_size=image_size, batch_size=batch_size)

    num_batches = len(anchor_dataloader)
    print(f"Number of batches in signature_dataloader: {num_batches}")

    # Train the Siamese network
    t.train_siamese_network(anchor_dataloader, positive_dataloader, negative_dataloader, desired_embedding_dim, num_epochs)


if __name__ == "__main__":
    main()

from preprocess import preprocess_data
from visualize import visualize_signature_images
import train as t
from siamese_net import SiameseNetwork


def main():
    train_anchor_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/train/anchor"
    train_positive_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/train/positive"
    train_negative_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/train/negative"

    validate_anchor_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/validate/anchor"
    validate_positive_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/validate/positive"
    validate_negative_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/validate/negative"

    test_anchor_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/test/anchor"
    test_positive_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/test/positive"
    test_negative_folder_path = "/Users/prattoymajumder/PycharmProjects/handwritten-signature-verification/triplet_dataset/test/negative"

    image_size = (128, 128)
    batch_size = 1
    num_epochs = 2
    desired_embedding_dim = 128

    # preprocess data
    train_anchor_dataloader = preprocess_data(train_anchor_folder_path, image_size=image_size, batch_size=batch_size)
    train_positive_dataloader = preprocess_data(train_positive_folder_path, image_size=image_size, batch_size=batch_size)
    train_negative_dataloader = preprocess_data(train_negative_folder_path, image_size=image_size, batch_size=batch_size)

    validate_anchor_dataloader = preprocess_data(validate_anchor_folder_path, image_size=image_size, batch_size=batch_size)
    validate_positive_dataloader = preprocess_data(validate_positive_folder_path, image_size=image_size, batch_size=batch_size)
    validate_negative_dataloader = preprocess_data(validate_negative_folder_path, image_size=image_size, batch_size=batch_size)

    test_anchor_dataloader = preprocess_data(test_anchor_folder_path, image_size=image_size, batch_size=batch_size)
    test_positive_dataloader = preprocess_data(test_positive_folder_path, image_size=image_size, batch_size=batch_size)
    test_negative_dataloader = preprocess_data(test_negative_folder_path, image_size=image_size, batch_size=batch_size)
    test_dataloader = zip(test_anchor_dataloader, test_positive_dataloader, test_negative_dataloader)

    num_batches = len(train_anchor_dataloader)
    print(f"Number of batches in signature_dataloader: {num_batches}")

    siamese_net = SiameseNetwork(embedding_dim=desired_embedding_dim, num_heads=4, num_layers=2)  # Use the SiameseNetwork with Transformer

    # Train the Siamese network
    t.train_siamese_network(siamese_net, train_anchor_dataloader, train_positive_dataloader, train_negative_dataloader, num_epochs)

    # Validate the Siamese network

    # Test the Siamese network
    t.test_siamese_network(siamese_net, test_dataloader)

if __name__ == "__main__":
    main()

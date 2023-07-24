from preprocess import preprocess_data
from transformer import extract_embeddings
from visualize import visualize_signature_images


def main():
    data_folder = "/Users/prattoymajumder/Downloads/archive (1)/sample_Signature/sample_Signature/forged"
    image_size = (256, 256)
    batch_size = 32

    signature_dataloader = preprocess_data(data_folder, image_size=image_size, batch_size=batch_size)

    # to visualize if the data has been preprocessed or not
    # visualize_signature_images(signature_dataloader)

    # Assuming 'signature_dataloader' contains preprocessed signature images
    for batch_images in signature_dataloader:
        signature_embeddings = extract_embeddings(batch_images)

    # The 'signature_embeddings' tensor now contains the learned representations from the Transformer.
    # Each row in 'signature_embeddings' corresponds to an embedding representation for a signature image.


if __name__ == "__main__":
    main()

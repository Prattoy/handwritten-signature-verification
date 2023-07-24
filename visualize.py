import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def visualize_signature_images(dataloader):
    # Get a batch of signature images from the DataLoader
    batch_images = next(iter(dataloader))

    # Unnormalize the images to visualize them
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2]),
        transforms.ToPILImage(),
    ])

    # Visualize a random sample of images from the batch
    num_samples_to_visualize = 4
    fig, axes = plt.subplots(1, num_samples_to_visualize, figsize=(12, 3))

    for i in range(num_samples_to_visualize):
        image = unnormalize(batch_images[i].unsqueeze(0).squeeze(0))  # Add and remove batch dimension
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')

    plt.show()

if __name__ == "__main__":
    # Assuming you have already loaded the 'signature_dataloader' in 'main.py'
    from main import preprocess_data

    data_folder = "/Users/prattoymajumder/Downloads/archive (1)/sample_Signature/sample_Signature/forged"
    image_size = (128, 128)
    batch_size = 32

    signature_dataloader = preprocess_data(data_folder, image_size=image_size, batch_size=batch_size)

    visualize_signature_images(signature_dataloader)

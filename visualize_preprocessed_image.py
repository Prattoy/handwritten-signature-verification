import matplotlib.pyplot as plt
import torchvision.utils as vutils


def visualize_batch(data_loader, idx):
    # Get the first batch from the dataloader
    batch = next(iter(data_loader))
    images = batch[idx]  # Assuming images are in the first index of the batch

    # Create a grid of images
    grid = vutils.make_grid(images, nrow=8, padding=2, normalize=True)

    # Display the grid of images
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
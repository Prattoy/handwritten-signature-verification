import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SignatureDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_list = os.listdir(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = os.path.join(self.data_folder, self.image_list[idx])

        if image_name.endswith(".png"):
            image = Image.open(image_name)

            # Convert the grayscale image to 1-channel grayscale
            image = image.convert("L")

            if self.transform:
                image = self.transform(image)

            return image


def preprocess_data(data_folder, image_size=(128, 128), batch_size=32, test=False):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),  # resizes images
        transforms.ToTensor(),  # converts to tensor
        # transforms.Normalize(mean=[0.5], std=[0.5]),  # normalization
    ])

    signature_dataset = SignatureDataset(data_folder, transform=data_transform)
    # Filter out None elements from the signature_dataset
    signature_dataset = [signatures for signatures in signature_dataset if signatures is not None]

    if test:
        # batch_size = len(signature_dataset)
        signature_dataloader = signature_dataset
    else:
        signature_dataloader = DataLoader(signature_dataset, batch_size=batch_size, shuffle=False)

    return signature_dataloader

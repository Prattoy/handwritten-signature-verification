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
        image = Image.open(image_name)

        # Convert the grayscale image to RGB
        image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform:
            image = self.transform(image)

        return image

def preprocess_data(data_folder, image_size=(128, 128), batch_size=32):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Corrected normalization
    ])

    signature_dataset = SignatureDataset(data_folder, transform=data_transform)
    signature_dataloader = DataLoader(signature_dataset, batch_size=batch_size, shuffle=True)

    return signature_dataloader

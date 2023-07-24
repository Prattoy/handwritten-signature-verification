import torch
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel

def extract_embeddings(signature_images, model_path="google/vit-base-patch16-224-in21k"):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
    model = ViTModel.from_pretrained(model_path)
    model.eval()

    # Normalize the images to the range [0, 1]
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    normalized_images = [transform(image) for image in signature_images]

    with torch.no_grad():
        inputs = feature_extractor(normalized_images, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    return embeddings

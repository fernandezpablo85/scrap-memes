import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from typing import Tuple
from PIL import Image
import os


def load_model():
    model = resnet18(weights=None)  # No preloaded weights
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 32),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(32, 2),  # Binary classification
    )
    model.load_state_dict(
        torch.load(
            os.path.join(os.path.dirname(__file__), "model_weights.pth"),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    model.eval()  # Set to evaluation mode
    return model


model = load_model()


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to match ResNet input
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


def classify(image: Image) -> Tuple[str, float]:
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
    class_labels = {0: "Frog Meme 🐸", 1: "Not Meme"}
    return class_labels[predicted.item()], probabilities

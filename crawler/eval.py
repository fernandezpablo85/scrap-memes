import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import logger

logger.setup_logging()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    return accuracy, precision, recall, f1


def main():
    # Load the model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2),
    )

    # Load the weights
    model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    model.eval()  # Set to evaluation mode
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ResNet-50 input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Pretrained model normalization
        ]
    )

    # Load the test dataset
    test_dataset = datasets.ImageFolder(
        "dataset/augmented_validation", transform=transform
    )  # yea this should be a different dataset but we're just classifying memes so
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":
    main()

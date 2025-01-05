import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

EPOCHS = 15
PATIENCE = 3


def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(EPOCHS):  # Number of epochs
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        model.train()
        if epochs_without_improvement > PATIENCE:
            break

    return best_model_state, best_val_loss


def main():
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

    # Load the datasets
    train_dataset = datasets.ImageFolder("dataset/augmented_train", transform=transform)
    val_dataset = datasets.ImageFolder(
        "dataset/augmented_validation", transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize the model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head with a minimal one
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 32),  # Very small dense layer
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(32, 2),  # Final output layer for 2 classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    class_weights = torch.tensor([2.0, 1.0], dtype=torch.float).to(
        device
    )  # Adjust weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Train the model
    best_state, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, device
    )

    # Save the model weights
    torch.save(best_state, "model_weights.pth")


if __name__ == "__main__":
    main()

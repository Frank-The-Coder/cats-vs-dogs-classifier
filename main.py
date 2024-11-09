import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_data_loader
from src.model import SimpleCNN
from src.train import train
from src.evaluate import evaluate
from src.plot import plot_training_curve
from torchvision import transforms

def main():
    # Initialize model, loss, and optimizer
    model = SimpleCNN()
    criterion = nn.BCEWithLogitsLoss()
    
    # New hyperparameters
    learning_rate = 0.001   # Optimized learning rate
    batch_size = 32         # Adjusted batch size
    num_epochs = 10         # Epochs kept at 10 for initial tests
    target_classes = ['cat', 'dog']
    
    # Apply data augmentation to the training set
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test data transform (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load data with augmentation for training
    train_loader, val_loader, test_loader, _ = get_data_loader(target_classes, batch_size, transform_train, transform_test)

    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # Train model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Evaluate model
    test_err, test_loss = evaluate(model, test_loader, criterion)
    print(f'Test Error: {test_err}, Test Loss: {test_loss}')

if __name__ == '__main__':
    main()

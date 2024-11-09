import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision

def get_relevant_indices(dataset, classes, target_classes):
    """Return indices for dataset items matching target classes"""
    indices = []
    for i in range(len(dataset)):
        label_index = dataset[i][1]
        label_class = classes[label_index]
        if label_class in target_classes:
            indices.append(i)
    return indices

def get_data_loader(target_classes, batch_size, transform_train, transform_test):
    """Loads images and splits data into training, validation, and testing with transformation"""
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Training data with transformations
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # Get relevant indices for target classes
    relevant_indices = get_relevant_indices(trainset, classes, target_classes)
    np.random.seed(1000)
    np.random.shuffle(relevant_indices)
    split = int(len(relevant_indices) * 0.8)
    relevant_train_indices, relevant_val_indices = relevant_indices[:split], relevant_indices[split:]
    
    # Create samplers and data loaders
    train_sampler = SubsetRandomSampler(relevant_train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(relevant_val_indices)
    val_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=val_sampler)
    
    # Test data with transformations
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    relevant_test_indices = get_relevant_indices(testset, classes, target_classes)
    test_sampler = SubsetRandomSampler(relevant_test_indices)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader, classes

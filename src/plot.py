import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve(train_err, val_err, train_loss, val_loss):
    epochs = range(1, len(train_err) + 1)
    plt.figure()
    plt.plot(epochs, train_err, label='Train Error')
    plt.plot(epochs, val_err, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

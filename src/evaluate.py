import torch

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_err, total_epoch = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            labels = (labels > 0).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            err = (outputs > 0.5).long().squeeze() != labels
            total_err += err.sum().item()
            total_loss += loss.item()
            total_epoch += len(labels)
    return total_err / total_epoch, total_loss / len(loader)

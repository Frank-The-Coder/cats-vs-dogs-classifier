import torch

def normalize_label(labels):
    max_val, min_val = torch.max(labels), torch.min(labels)
    return (labels - min_val) / (max_val - min_val)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            labels = normalize_label(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            # Accumulate train loss and correct predictions
            running_train_loss += loss.item()
            predictions = (outputs > 0).long().squeeze()
            correct_train += (predictions == labels.long()).sum().item()
            total_train += labels.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_error = 1 - (correct_train / total_train)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = normalize_label(labels)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().view(-1, 1))
                running_val_loss += loss.item()

                # Accumulate validation correct predictions
                predictions = (outputs > 0).long().squeeze()
                correct_val += (predictions == labels.long()).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_error = 1 - (correct_val / total_val)

        # Print results for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}')

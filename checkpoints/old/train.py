# Train.py: Contains the training function for our model.
# It manages training and validation loops, tracks performance, and saves the best model.

import torch
import torch.nn as nn
import time, copy

def train_model(model, dataloaders, criterion, optimizer, scheduler=None,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                num_epochs=25, save_best_model_path=None):
    """
    Train the model and return the version with the best validation accuracy.

    Parameters:
      - model: The neural network to train.
      - dataloaders: A dictionary with 'train' and 'val' DataLoader objects.
      - criterion: Loss function (e.g., CrossEntropyLoss).
      - optimizer: Optimizer to update model weights.
      - scheduler: (Optional) Learning rate scheduler.
      - device: The device to use (GPU if available, otherwise CPU).
      - num_epochs: Number of training epochs.
      - save_best_model_path: Path to save the best model weights.

    Returns:
      - model: The trained model with the best validation accuracy.
    """
    print("Starting training...\n")
    since = time.time()  # Start timer

    best_model_wts = copy.deepcopy(model.state_dict())  # Save initial model weights.
    best_acc = 0.0  # Best validation accuracy seen so far.

    model.to(device)  # Move model to the proper device.

    # Loop over each epoch.
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # Each epoch consists of a training and validation phase.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode (enables dropout, etc.)
            else:
                model.eval()   # Set model to evaluation mode (disables dropout)

            running_loss = 0.0  # To accumulate loss over the epoch.
            running_corrects = 0  # To accumulate number of correct predictions.
            total = 0  # Total number of samples processed.

            # Iterate over the data for the current phase.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # Move inputs to device.
                labels = labels.to(device)  # Move labels to device.

                optimizer.zero_grad()  # Zero out gradients from previous step.

                # Enable gradient tracking only in training phase.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Forward pass: compute predictions.
                    loss = criterion(outputs, labels)  # Calculate loss.
                    _, preds = torch.max(outputs, 1)  # Get predicted classes.

                    # Backpropagation only during training.
                    if phase == 'train':
                        loss.backward()   # Compute gradients.
                        optimizer.step()  # Update weights.

                # Accumulate loss and correct predictions.
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            # Calculate average loss and accuracy for this phase.
            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # If validation accuracy improved, save the model weights.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_best_model_path:
                    torch.save(best_model_wts, save_best_model_path)
        
        # If a scheduler is provided, update the learning rate.
        if scheduler:
            scheduler.step()
        print()

    # Display training time and best accuracy.
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load the best model weights and return the model.
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # This script is meant to be imported, not run directly.
    pass

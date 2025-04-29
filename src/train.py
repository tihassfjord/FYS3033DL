"""
Neural Network Training Module

This module provides the functionality to train deep learning models with
comprehensive tracking of metrics, early stopping, learning rate scheduling,
and visualization of training progress.

Key features:
- Training with multiple phases (train/validation)
- Progress tracking with tqdm
- Model checkpointing (best model saving)
- Early stopping to prevent overfitting
- Learning rate scheduling compatibility
- Training metrics visualization
- CSV logging of training metrics

Example usage:
    model = YourModel()
    dataloaders = {'train': train_loader, 'val': val_loader}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    trained_model, model_name = train_model(
        model, dataloaders, criterion, optimizer,
        scheduler=scheduler, num_epochs=20, patience=5
    )
"""

from tqdm import tqdm
import torch
import time
import os
import matplotlib.pyplot as plt 
from datetime import datetime
import copy


def ensure_dir(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, device='cuda',
                num_epochs=10, model_name='model', patience=10):
    """
    Train a PyTorch model with comprehensive progress tracking and visualization.
    
    The function handles the complete training workflow including:
    - Training and validation phases
    - Early stopping based on validation accuracy
    - Learning rate scheduling
    - Model checkpointing (saves best model)
    - Progress visualization
    - Metrics logging
    
    Args:
        model: PyTorch model to train
        dataloaders: Dictionary with 'train' and 'val' data loaders
        criterion: Loss function
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Maximum number of training epochs
        model_name: Base name for saved model files
        patience: Number of epochs to wait for improvement before early stopping
    
    Returns:
        tuple: (trained_model, model_name) - The trained model and its unique name
    """
    # Create unique model name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{model_name}_{timestamp}"

    # Initialize tracking variables
    start_time = time.time()
    best_acc = 0.0
    best_loss = float('inf')  # Initialize best loss for comparison
    best_model_wts = None
    best_epoch = 0  # Track which epoch had the best val accuracy

    # Lists to store training metrics
    train_losses = []
    val_accuracies = []
    val_losses = []
    lr_rates = []

    # Setup progress bar
    total_steps = num_epochs * sum(len(loader) for loader in dataloaders.values())
    global_loop = tqdm(total=total_steps, desc=f"Training {model_name}", unit='batch')
    
    # Check if we're using a DataPrefetcher (which already handles device movement)
    using_prefetcher = hasattr(dataloaders['train'], 'stream') if 'train' in dataloaders else False

    # Training loop
    for epoch in range(num_epochs):
        # Track the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_rates.append(current_lr)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Set model to training mode or evaluation mode
            model.train() if phase == 'train' else model.eval()

            # Initialize metrics for this phase
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Check if we're using a DataPrefetcher (which already handles device movement)
            using_prefetcher = hasattr(dataloaders['train'], 'stream') if 'train' in dataloaders else False
            
            # And in the data loading section:
            for data in dataloaders[phase]:
                # Handle different return types (prefetcher vs standard dataloader)
                if using_prefetcher:
                    # DataPrefetcher already puts data on the right device
                    inputs, labels = data
                else:
                    # Standard DataLoader - we need to move data to device ourselves
                    inputs, labels = data
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass - track gradients only during training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        # Step the scheduler if it's not ReduceLROnPlateau
                        # (ReduceLROnPlateau needs validation metrics to step)
                        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step()
                            
                # Accumulate batch statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Update progress bar with batch info
                batch_acc = (torch.sum(preds == labels.data).item() / inputs.size(0))
                global_loop.set_postfix({
                    'epoch': f"{epoch+1}/{num_epochs}",
                    'phase': phase,
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
                global_loop.update(1)

            # Calculate epoch statistics
            if total_samples > 0:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples
            else:
                print(f"Warning: No samples in {phase} phase for epoch {epoch+1}. Using default values.")
                epoch_loss = float('inf') if phase == 'train' else best_loss
                epoch_acc = 0.0 if phase == 'train' else best_acc

            # Store phase metrics
            if phase == 'val':
                val_accuracies.append(epoch_acc.item())
                val_losses.append(epoch_loss)

                # Check if this is the best validation accuracy
                # Also use loss as a tiebreaker when accuracies are equal
                if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch  # Update the best epoch tracker
            else:
                # Store training metrics
                train_losses.append(epoch_loss)
                
            # Force memory cleanup after each phase
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Step the ReduceLROnPlateau scheduler after validation
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use the monitored metric based on the scheduler mode
                metric = epoch_loss if scheduler.mode == 'min' else epoch_acc
                scheduler.step(metric)
            
        # Early stopping check
        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            print(f"Best validation accuracy was {best_acc:.4f} at epoch {best_epoch+1}.")
            break

    # Finalize training
    global_loop.close()
    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f}s")

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Save model files
    ensure_dir("models")
    # Uncomment to save final model weights as well
    # torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    torch.save(best_model_wts, f"models/{model_name}_best.pth")

    # Save metrics to CSV
    ensure_dir("logs")
    with open(f"logs/{model_name}_metrics.csv", 'w') as f:
        f.write("epoch,train_loss,val_loss,val_acc,lr\n")
        for i in range(len(val_accuracies)):
            lr = lr_rates[i]
            f.write(f"{i+1},{train_losses[i]:.4f},{val_losses[i]:.4f},{val_accuracies[i]:.4f},{lr:.8f}\n")

    # Generate and save visualization of training metrics
    visualize_training_metrics(
        train_losses, val_losses, val_accuracies, lr_rates, 
        epochs=len(val_accuracies), model_name=model_name
    )

    return model, model_name


def visualize_training_metrics(train_losses, val_losses, val_accuracies, lr_rates, epochs, model_name):
    """
    Create and save visualization of training metrics.
    
    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        val_accuracies (list): Validation accuracy per epoch
        lr_rates (list): Learning rates per epoch
        epochs (int): Number of epochs completed
        model_name (str): Name for saved plot file
    """
    # Plotting training curves: 3 subplots
    plt.figure(figsize=(18, 5))  # make room for 3 panels

    # Subplot 1: Training Loss vs Epoch
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # Subplot 2: Validation Accuracy vs Epoch
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs+1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.legend()

    # Subplot 3: Learning Rate and Validation Loss vs Epoch together
    plt.subplot(1, 3, 3)
    epoch_range = range(1, epochs+1)
    
    # Primary y-axis: Learning Rate
    ax1 = plt.gca()
    ax1.plot(epoch_range, lr_rates[:epochs], marker='o', color='tab:blue', label='Learning Rate')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Secondary y-axis: Validation Loss
    ax2 = ax1.twinx()
    ax2.plot(epoch_range, val_losses, marker='o', color='tab:red', label='Validation Loss')
    ax2.set_ylabel('Validation Loss', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Learning Rate & Validation Loss')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    ensure_dir("plots")
    plt.savefig(f"plots/{model_name}_training.png")
    plt.show()


if __name__ == '__main__':
    """
    Example usage script.
    
    To use this module directly, uncomment and modify the example code below:
    
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import models
    
    # Define your model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Configure training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train model
    trained_model, model_name = train_model(
        model, dataloaders, criterion, optimizer, 
        scheduler=scheduler, device=device, 
        num_epochs=20, patience=5
    )
    """
    
    # Placeholder for example execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This module provides training functionality to be imported. Device: {device}")
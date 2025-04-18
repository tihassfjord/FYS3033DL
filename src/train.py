# the contents of a file named train.py
# It contains a training loop, evaluation function, etc.

from tqdm import tqdm
import torch
import time
import os
import matplotlib.pyplot as plt 
from datetime import datetime


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, device='cuda',
                num_epochs=10, model_name='model', patience=10):
    """
    Trains the model using provided dataloaders and configuration.
    Includes early stopping based on validation accuracy.
    """

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{model_name}_{timestamp}"

    start_time = time.time()
    best_acc = 0.0
    best_model_wts = None
    best_epoch = 0  # Track which epoch had the best val accuracy

    # Lists to store training metrics
    train_losses = []
    val_accuracies = []
    val_losses = []
    lr_rates = []

    total_steps = num_epochs * sum(len(loader) for loader in dataloaders.values())
    global_loop = tqdm(total=total_steps, desc=f"Training {model_name}", unit='batch')

    for epoch in range(num_epochs):
        
        # logs the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_rates.append(current_lr)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                batch_acc = (torch.sum(preds == labels.data).item() / inputs.size(0))
                global_loop.set_postfix({
                    'epoch': f"{epoch+1}/{num_epochs}",
                    'phase': phase,
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
                global_loop.update(1)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            if phase == 'val':
                val_accuracies.append(epoch_acc.item())
                val_losses.append(epoch_loss)

                # Check if this is the best validation accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    best_epoch = epoch  # Reset patience counter
                # else:
                    # Print patience remaining if no improvement
                    # tqdm.write(f"Patience remaining: {patience - (epoch - best_epoch)}")


            else:
                train_losses.append(epoch_loss)

        if scheduler:
            scheduler.step()

        # Early stopping check
        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            print(f"Best validation accuracy was {best_acc:.4f} at epoch {best_epoch+1}.")
            break

    global_loop.close()
    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f}s")

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Save model files
    ensure_dir("models")
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    torch.save(best_model_wts, f"models/{model_name}_best.pth")

    # Save metrics
    ensure_dir("logs")
    with open(f"logs/{model_name}_metrics.csv", 'w') as f:
        f.write("epoch,train_loss,val_loss,val_acc,lr\n")
        for i in range(len(val_accuracies)):
            lr = lr_rates[i]
            f.write(f"{i+1},{train_losses[i]:.4f},{val_losses[i]:.4f},{val_accuracies[i]:.4f},{lr:.8f}\n")


    # Eval metrics
    # Plotting training curves: 3 subplots now
    plt.figure(figsize=(18, 5))  # make room for 3 panels

    # Subplot 1: Training Loss vs Epoch
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.grid(True); plt.legend()

    # Subplot 2: Validation Accuracy vs Epoch
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy')
    plt.grid(True); plt.legend()

    # Subplot 3: Learning Rate vs Validation Loss
    plt.subplot(1, 3, 3)
    plt.plot(lr_rates, val_losses, marker='o', label='LR vs Val Loss')
    plt.xlabel('Learning Rate'); plt.ylabel('Validation Loss'); plt.title('LR vs Validation Loss')
    plt.grid(True); plt.legend()

    plt.tight_layout()
    ensure_dir("plots")
    plt.savefig(f"plots/{model_name}_training.png")
    plt.show()


    return model


# def train_model(model, dataloaders, criterion, optimizer, scheduler=None, device='cuda', num_epochs=10, model_name='model'):
   
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     model_name = f"{model_name}_{timestamp}"

#     start_time = time.time()
#     best_acc = 0.0
#     best_model_wts = None

#     # Lists to store training loss and validation accuracy for each epoch
#     train_losses = []
#     val_accuracies = []
#     val_losses = []

#     # Total steps = #epochs * (len(train) + len(val))
#     total_steps = num_epochs * sum(len(loader) for loader in dataloaders.values())
#     global_loop = tqdm(total=total_steps, desc=f"Training {model_name}", unit='batch')

#     for epoch in range(num_epochs):

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             model.train() if phase == 'train' else model.eval()

#             running_loss = 0.0
#             running_corrects = 0
#             total_samples = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     _, preds = torch.max(outputs, 1)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#                 total_samples += inputs.size(0)

#                 # Update global progress bar
#                 batch_acc = (torch.sum(preds == labels.data).item() / inputs.size(0))
#                 global_loop.set_postfix({
#                     'epoch': f"{epoch+1}/{num_epochs}",
#                     'phase': phase,
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{batch_acc:.4f}'
#                 })
#                 global_loop.update(1)

#             epoch_loss = running_loss / total_samples
#             epoch_acc = running_corrects.double() / total_samples

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = model.state_dict()
                

#             if phase == 'train':
#                 train_losses.append(epoch_loss)
#             else:
#                 val_accuracies.append(epoch_acc.item())
#                 val_losses.append(epoch_loss)

#         if scheduler:
#             scheduler.step()

#     global_loop.close()
#     print(f"\nBest Val Accuracy: {best_acc:.4f}")
#     print(f"Total training time: {time.time() - start_time:.2f}s")

#     if best_model_wts is not None:
#         model.load_state_dict(best_model_wts)

#     ensure_dir("models")
#     torch.save(model.state_dict(), f"models/{model_name}_final.pth")
#     torch.save(best_model_wts, f"models/{model_name}_best.pth")

#     ensure_dir("logs")
#     with open(f"logs/{model_name}_metrics.csv", 'w') as f:
#         f.write("epoch,train_loss,val_loss,val_acc\n")
#         for i in range(num_epochs):
#             val_loss = val_losses[i] if i < len(val_losses) else 'N/A'
#             f.write(f"{i+1},{train_losses[i]:.4f},{val_loss},{val_accuracies[i]:.4f}\n")

#     # Plotting training curve
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
#     plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True); plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', label='Validation Accuracy')
#     plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.grid(True); plt.legend()

#     plt.tight_layout()
#     ensure_dir("plots")
#     plt.savefig(f"plots/{model_name}_training.png")
#     plt.show()

#     return model

if __name__ == '__main__':
    
    # Example usage
    # Assuming you have a model, dataloaders, criterion, optimizer defined
    # model = YourModel()
    # dataloaders = {'train': train_loader, 'val': val_loader}
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trained_model = train_model(model, dataloaders, criterion, optimizer, device=device, num_epochs=10)
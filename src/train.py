# the contents of a file named train.py
# It contains a training loop, evaluation function, etc.

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, device='cuda', num_epochs=10, model_name='model'):
    start_time = time.time()
    best_acc = 0.0
    best_model_wts = None
    # Lists to store training loss and validation accuracy for each epoch
    train_losses = []
    val_accuracies = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            ''' applying a TQDM progress bar to the training loop, ordinary forloop is commented out '''
            loop = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

                # Show live metrics
                batch_acc = (torch.sum(preds == labels.data).item() / inputs.size(0))
                loop.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_acc:.4f}')
            
            # for inputs, labels in dataloaders[phase]:
            #     inputs = inputs.to(device)
            #     labels = labels.to(device)

            #     optimizer.zero_grad()

            #     with torch.set_grad_enabled(phase == 'train'):
            #         outputs = model(inputs)
            #         loss = criterion(outputs, labels)
            #         _, preds = torch.max(outputs, 1)

            #         if phase == 'train':
            #             loss.backward()
            #             optimizer.step()

            #     running_loss += loss.item() * inputs.size(0)
            #     running_corrects += torch.sum(preds == labels.data)
            #     total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            # print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            


            # Update best model weights based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print(f"New best model found at epoch {epoch+1} with accuracy {epoch_acc:.4f}")
                # torch.save(best_model_wts, f'{model_name}_best.pth')  # <- Saves best model weights
                


                
            # Store metrics for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
            elif phase == 'val':
                val_accuracies.append(epoch_acc.item())
                val_losses.append(epoch_loss)

        if scheduler is not None:
            scheduler.step()
        print()

    print(f"Best val Acc: {best_acc:.4f}")
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # ending time count
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")


    # Save final model
    ensure_dir("models")
    torch.save(model.state_dict(), f"models/{model_name}_final.pth") # <- Saves final model weights

    # Save best model weights
    torch.save(best_model_wts, f"models/{model_name}_best.pth")

    # Save training metrics to CSV
    ensure_dir("logs")
    with open(f"logs/{model_name}_metrics.csv", 'w') as f:
        f.write("epoch,train_loss,val_loss,val_acc\n")
        for i in range(num_epochs):
            val_loss = val_losses[i] if i < len(val_losses) else 'N/A'
            f.write(f"{i+1},{train_losses[i]:.4f},{val_loss},{val_accuracies[i]:.4f}\n")
            

    
    # Plot training loss and validation accuracy over epochs
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save training curves as plot
    ensure_dir("plots")
    plt.savefig(f"plots/{model_name}_training.png")

    plt.show()

    return model


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
    
    

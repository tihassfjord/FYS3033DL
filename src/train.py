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

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, device='cuda', num_epochs=10, model_name='model'):
   
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{model_name}_{timestamp}"

    start_time = time.time()
    best_acc = 0.0
    best_model_wts = None

    # Lists to store training loss and validation accuracy for each epoch
    train_losses = []
    val_accuracies = []
    val_losses = []

    # Total steps = #epochs * (len(train) + len(val))
    total_steps = num_epochs * sum(len(loader) for loader in dataloaders.values())
    global_loop = tqdm(total=total_steps, desc=f"Training {model_name}", unit='batch')

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
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

                # Update global progress bar
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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc.item())
                val_losses.append(epoch_loss)

        if scheduler:
            scheduler.step()

    global_loop.close()
    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f}s")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    ensure_dir("models")
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    torch.save(best_model_wts, f"models/{model_name}_best.pth")

    ensure_dir("logs")
    with open(f"logs/{model_name}_metrics.csv", 'w') as f:
        f.write("epoch,train_loss,val_loss,val_acc\n")
        for i in range(num_epochs):
            val_loss = val_losses[i] if i < len(val_losses) else 'N/A'
            f.write(f"{i+1},{train_losses[i]:.4f},{val_loss},{val_accuracies[i]:.4f}\n")

    # Plotting training curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.grid(True); plt.legend()

    plt.tight_layout()
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
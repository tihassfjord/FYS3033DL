# src/evaluation.py
# Contains all functions related to evaluation of the model.
# This includes plotting confusion matrices, classification reports, and entropy histograms.

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_multiclass_roc(model, dataloader, device, num_classes=3, class_names=None, model_name="model"):
    """
    Plots and saves a multiclass ROC curve for classification models.
    Assumes model outputs logits (before softmax).
    """
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            y_score.extend(outputs)
            y_true.extend(labels.numpy())

    y_score = np.array(y_score)
    y_true = np.array(y_true)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    ensure_dir("plots")
    save_path = f"plots/ROC_{model_name}.png"
    plt.savefig(save_path)
    plt.show()

    print(f"[ROC] Saved ROC plot to: {save_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """
    Plots a bar chart of accuracy per class.
    """
    class_accs = []
    for cls in range(len(class_names)):
        mask = np.array(y_true) == cls
        acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
        class_accs.append(acc)

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, class_accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.grid(True)
    if save_path:
        ensure_dir("plots")
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(model, dataloader, device, class_names=None, save_path=None):
    ''' 
    Plots a confusion matrix for the model predictions on the given dataloader.
    '''
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# def classification_summary(model, dataloader, device='cuda', class_names=None):
#     ''' 
#     Prints a classification report including precision, recall, and F1-score.
#     '''
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())

    
#     print(classification_report(y_true, y_pred, target_names=class_names))
#     return y_true, y_pred  # for custom plots

def classification_summary(model, dataloader, criterion, device='cuda', class_names=None):
    model.eval()
    y_true, y_pred = [], []
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total

    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))
    return avg_loss, accuracy, y_true, y_pred  # for custom plots


def plot_entropy_hist(entropies_known, entropies_unknown, bins=30): 
    """
    Plots two histograms on the same figure.
    """
    plt.figure()
    plt.hist(entropies_known, bins=bins, alpha=0.5, label='Known Class Entropy')
    plt.hist(entropies_unknown, bins=bins, alpha=0.5, label='Unknown Class Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Entropy Histograms for Known vs. Unknown')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # Assuming you have a trained model and a dataloader ready
    # model = ...  # Your trained model
    # dataloader = ...  # Your dataloader for validation/test set
    # class_names = ['Class1', 'Class2', 'Class3']  # Replace with your actual class names

    # plot_confusion_matrix(model, dataloader, device='cuda', class_names=class_names, save_path='confusion_matrix.png')
    # classification_summary(model, dataloader, device='cuda', class_names=class_names)
    pass
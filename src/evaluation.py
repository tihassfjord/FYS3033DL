# src/evaluation.py
# Contains all functions related to evaluation of the model.
# This includes plotting confusion matrices, classification reports, and entropy histograms.

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

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

def classification_summary(model, dataloader, device='cuda', class_names=None):
    ''' 
    Prints a classification report including precision, recall, and F1-score.
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
    
    print(classification_report(y_true, y_pred, target_names=class_names))
    return y_true, y_pred  # for custom plots

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
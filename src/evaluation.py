"""
Model Evaluation Module

This module provides comprehensive tools for evaluating deep learning models,
particularly for classification tasks. It includes functions for generating
various evaluation metrics, visualizations, and performance analyses.

Key features:
- Classification performance metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- ROC curves for multi-class classification
- Per-class accuracy analysis
- Prediction visualization
- Entropy distribution analysis for uncertainty quantification
- Calibration analysis

Example usage:
    model = YourModel()
    dataloader = test_dataloader
    class_names = ['Cat', 'Dog', 'Bird']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get model performance summary
    loss, acc, y_true, y_pred = classification_summary(
        model, dataloader, criterion, device, class_names
    )

    # Visualize confusion matrix
    plot_confusion_matrix(model, dataloader, device, class_names)

    # Generate ROC curves
    plot_multiclass_roc(model, dataloader, device, num_classes=3, class_names=class_names)
"""

import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm


def ensure_dir(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def collect_predictions(model, dataloader, device):
    """
    Collect true labels and model predictions from a dataloader.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader for the dataset to evaluate
        device (torch.device): The device to run inference on
    
    Returns:
        tuple: (y_true, y_pred, y_scores) where:
            - y_true is a list of true labels
            - y_pred is a list of predicted labels
            - y_scores is a numpy array of prediction scores (logits)
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Collecting predictions"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)


def plot_multiclass_roc(model, dataloader, device, num_classes=3, class_names=None, model_name="model", save_path=None):
    """
    Plots and saves a multiclass ROC curve for classification models.
    
    The Receiver Operating Characteristic (ROC) curve visualizes the trade-off
    between true positive rate and false positive rate at various thresholds.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader for the dataset to evaluate
        device (torch.device): The device to run inference on
        num_classes (int): Number of classes in the classification task
        class_names (list): List of class names for better visualization
        model_name (str): Model name for saving the plot
        save_path (str, optional): Path to save the plot
    
    Returns:
        dict: Dictionary containing AUC scores for each class
    """
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Computing ROC curves"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            y_score.extend(outputs)
            y_true.extend(labels.numpy())

    y_score = np.array(y_score)
    y_true = np.array(y_true)

    # Binarize the labels for one-vs-rest ROC calculation
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # Calculate ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot the ROC curves
    plt.figure(figsize=(10, 6))
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color='navy', linestyle=':', linewidth=2)
    
    # Plot ROC curve for each class
    for i in range(num_classes):
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

    # Plot the diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-Class Classification')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = f"plots/ROC_{model_name}.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[ROC] Saved ROC plot to: {save_path}")
    return roc_auc


def plot_precision_recall_curves(model, dataloader, device, num_classes=3, class_names=None, model_name="model", save_path=None):
    """
    Plots and saves precision-recall curves for multi-class classification.
    
    The precision-recall curve shows the trade-off between precision and recall
    for different thresholds. It is particularly useful for imbalanced datasets.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader for the dataset to evaluate
        device (torch.device): The device to run inference on
        num_classes (int): Number of classes in the classification task
        class_names (list): List of class names for better visualization
        model_name (str): Model name for saving the plot
        save_path (str, optional): Path to save the plot
    
    Returns:
        dict: Dictionary containing average precision scores for each class
    """
    # Collect predictions
    y_true, _, y_scores = collect_predictions(model, dataloader, device)
    
    # Binarize the labels for one-vs-rest precision-recall calculation
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    # Calculate precision-recall curve and average precision for each class
    precision, recall, average_precision = {}, {}, {}
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    
    # Calculate micro-average precision-recall curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(
        y_true_bin.ravel(), y_scores.ravel())
    
    # Plot the precision-recall curves
    plt.figure(figsize=(10, 6))
    # Plot micro-average precision-recall curve
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average (AP = {average_precision["micro"]:.2f})',
             color='navy', linestyle=':', linewidth=2)
    
    # Plot precision-recall curve for each class
    for i in range(num_classes):
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(recall[i], precision[i],
                 label=f'{label} (AP = {average_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Multi-Class Classification')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = f"plots/PR_Curve_{model_name}.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[PR Curve] Saved precision-recall curve plot to: {save_path}")
    return average_precision


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """
    Plots a comparison of precision, recall, and F1-score for each class.
    
    Args:
        y_true (list): True class labels
        y_pred (list): Predicted class labels
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    # Calculate precision, recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)))
    
    # Also calculate accuracy per class
    class_accs = []
    for cls in range(len(class_names)):
        mask = np.array(y_true) == cls
        if np.sum(mask) > 0:  # Avoid division by zero
            acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
            class_accs.append(acc)
        else:
            class_accs.append(0.0)
    
    # Create a DataFrame for easier plotting
    import pandas as pd
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': class_accs
    })
    
    # Plot the metrics
    plt.figure(figsize=(12, 6))
    
    # Plot as grouped bar chart
    bar_width = 0.2
    index = np.arange(len(class_names))
    
    plt.bar(index - 1.5*bar_width, metrics_df['Precision'], bar_width, label='Precision')
    plt.bar(index - 0.5*bar_width, metrics_df['Recall'], bar_width, label='Recall')
    plt.bar(index + 0.5*bar_width, metrics_df['F1-Score'], bar_width, label='F1-Score')
    plt.bar(index + 1.5*bar_width, metrics_df['Accuracy'], bar_width, label='Accuracy')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(index, class_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = "plots/per_class_metrics.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[Per-Class Metrics] Saved metrics plot to: {save_path}")
    return metrics_df


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """
    Plots a bar chart of accuracy per class.
    
    Args:
        y_true (list): True class labels
        y_pred (list): Predicted class labels
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    
    Returns:
        numpy.ndarray: Array of per-class accuracies
    """
    class_accs = []
    for cls in range(len(class_names)):
        mask = np.array(y_true) == cls
        if np.sum(mask) > 0:  # Avoid division by zero
            acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
            class_accs.append(acc)
        else:
            class_accs.append(0.0)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accs, color='skyblue')
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.xlabel("Class")
    plt.title("Per-Class Accuracy")
    plt.grid(True, axis='y')
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Add overall accuracy as a horizontal line
    overall_acc = accuracy_score(y_true, y_pred)
    plt.axhline(y=overall_acc, color='r', linestyle='--', 
                label=f'Overall Accuracy: {overall_acc:.3f}')
    
    # Add balanced accuracy as another horizontal line
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    plt.axhline(y=balanced_acc, color='g', linestyle='--',
                label=f'Balanced Accuracy: {balanced_acc:.3f}')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = "plots/per_class_accuracy.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[Accuracy] Saved per-class accuracy plot to: {save_path}")
    return np.array(class_accs)


def plot_confusion_matrix(model, dataloader, device, class_names=None, normalize=False, save_path=None):
    """
    Plots a confusion matrix for the model predictions on the given dataloader.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader for the dataset to evaluate
        device (torch.device): The device to run inference on
        class_names (list, optional): List of class names for better visualization
        normalize (bool): Whether to normalize confusion matrix values
        save_path (str, optional): Path to save the plot
    
    Returns:
        numpy.ndarray: The confusion matrix
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Computing confusion matrix"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = " (Normalized)"
    else:
        fmt = 'd'
        title_suffix = ""
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=class_names
    )
    disp.plot(cmap='Blues', values_format=fmt)
    plt.title(f"Confusion Matrix{title_suffix}")
    plt.grid(False)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        norm_str = "_normalized" if normalize else ""
        save_path = f"plots/confusion_matrix{norm_str}.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[Confusion Matrix] Saved to: {save_path}")
    return cm


def plot_predictions(model, data_loader, class_names, device, num_samples=32, save_path=None):
    """
    Plot a grid of images with their predicted and true labels.
    
    Args:
        model (torch.nn.Module): The trained model
        data_loader (DataLoader): DataLoader for the dataset to inspect
        class_names (list): List of class names
        device (torch.device): The device to run inference on
        num_samples (int): Number of samples to display
        save_path (str, optional): Path to save the plot
    """
    model.eval()  # Set the model to evaluation mode
    
    # Get a batch of images and labels
    all_images = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            all_images.extend(inputs.cpu())
            all_labels.extend(labels.cpu())
            all_preds.extend(preds.cpu())
            all_probs.extend(confidences.cpu())
            
            if len(all_images) >= num_samples:
                break
    
    # Limit to the requested number of samples
    all_images = all_images[:num_samples]
    all_labels = all_labels[:num_samples]
    all_preds = all_preds[:num_samples]
    all_probs = all_probs[:num_samples]
    
    # Determine grid size based on number of samples
    grid_size = int(np.ceil(np.sqrt(len(all_images))))
    
    # Plot the images with predicted and true labels
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (img, label, pred, prob) in enumerate(zip(all_images, all_labels, all_preds, all_probs)):
        if i >= len(axes):  # Stop if there are more images than grid slots
            break
            
        img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0, 1]
        
        # Color-code the title based on prediction correctness
        if label == pred:
            color = 'green'  # Correct prediction
        else:
            color = 'red'    # Incorrect prediction
            
        axes[i].imshow(img)
        axes[i].axis('off')
        title = f"True: {class_names[label]}\nPred: {class_names[pred]}\nConf: {prob:.2f}"
        axes[i].set_title(title, fontsize=8, color=color)
    
    # Hide unused subplots
    for i in range(len(all_images), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Model Predictions", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = "plots/predictions_grid.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[Predictions] Saved predictions grid to: {save_path}")


def plot_calibration_curve(model, dataloader, device, num_classes, class_names=None, n_bins=10, save_path=None):
    """
    Plot reliability diagram (calibration curve) for the model.
    
    A well-calibrated model should have predicted probabilities that match
    the observed frequencies. This function plots this relationship.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader for the dataset to evaluate
        device (torch.device): The device to run inference on
        num_classes (int): Number of classes in the classification task
        class_names (list): List of class names for better visualization
        n_bins (int): Number of bins for the calibration curve
        save_path (str, optional): Path to save the plot
    
    Returns:
        tuple: (prob_true, prob_pred) containing the calibration curve data
    """
    # Collect predictions
    y_true, _, y_scores = collect_predictions(model, dataloader, device)
    
    # Convert logits to probabilities
    y_probs = torch.nn.functional.softmax(torch.tensor(y_scores), dim=1).numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Plot the calibration curve for each class
    for i in range(num_classes):
        # One-hot encode for current class
        y_true_bin = (y_true == i).astype(int)
        
        # Get the probability for the current class
        y_prob = y_probs[:, i]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true_bin, y_prob, n_bins=n_bins)
        
        # Plot calibration curve
        class_name = class_names[i] if class_names else f"Class {i}"
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                 label=f'{class_name}')
    
    # Plot the diagonal line (perfect calibration)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', 
             label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = "plots/calibration_curve.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[Calibration] Saved calibration curve to: {save_path}")
    return prob_true, prob_pred


def classification_summary(model, dataloader, criterion, device='cuda', class_names=None):
    """
    Generate a comprehensive classification summary for the model.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (DataLoader): DataLoader for the dataset to evaluate
        criterion: Loss function
        device (torch.device): The device to run inference on
        class_names (list, optional): List of class names for reporting
    
    Returns:
        tuple: (avg_loss, accuracy, y_true, y_pred) with evaluation metrics and data
    """
    model.eval()
    y_true, y_pred = [], []
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Store prediction probabilities for calibration analysis
    y_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating model"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predicted class and probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Accumulate metrics
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            # Store true labels, predictions and probabilities
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate overall metrics
    avg_loss = running_loss / total
    accuracy = correct / total
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Print classification report
    print("\n--- Classification Report ---")
    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    else:
        print(classification_report(y_true, y_pred))
    
    # Additional metrics
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Calculate per-class precision, recall, and F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None)
    
    # Create a summary DataFrame
    import pandas as pd
    summary_df = pd.DataFrame({
        'Class': class_names if class_names else [f'Class {i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    print("\n--- Per-Class Metrics ---")
    print(summary_df.to_string(index=False))
    
    return avg_loss, accuracy, y_true, y_pred


def plot_entropy_hist(entropies_known, entropies_unknown, save_path=None):
    """
    Plot histogram of entropy values for known vs unknown classes.
    
    Entropy is a measure of prediction uncertainty. Higher entropy values
    indicate less certain predictions. This function compares the entropy
    distributions between known and unknown classes.
    
    Args:
        entropies_known (array-like): Entropy values for known class predictions
        entropies_unknown (array-like): Entropy values for unknown class predictions
        save_path (str, optional): Path to save the plot
    
    Returns:
        float: Threshold value that could potentially separate known from unknown
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms with density curve
    sns.histplot(entropies_known, bins=50, kde=True, label='Known Classes', 
                 color='blue', stat='density', alpha=0.6)
    sns.histplot(entropies_unknown, bins=50, kde=True, label='Unknown Class', 
                 color='red', stat='density', alpha=0.6)
    
    # Calculate and display means
    mean_known = np.mean(entropies_known)
    mean_unknown = np.mean(entropies_unknown)
    std_known = np.std(entropies_known)
    std_unknown = np.std(entropies_unknown)

    plt.axvline(mean_known, color='blue', linestyle='--', 
                label=f'Mean Known: {mean_known:.3f} ± {std_known:.3f}')
    plt.axvline(mean_unknown, color='red', linestyle='--', 
                label=f'Mean Unknown: {mean_unknown:.3f} ± {std_unknown:.3f}')

    # Calculate a potential threshold for separating known/unknown
    # Using a simple approach: the midpoint between means
    threshold = (mean_known + mean_unknown) / 2
    plt.axvline(threshold, color='green', linestyle='-.',
                label=f'Potential Threshold: {threshold:.3f}')

    plt.xlabel('Prediction Entropy')
    plt.ylabel('Density')
    plt.title('Entropy Distribution: Known vs Unknown Classes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else "plots")
        plt.savefig(save_path)
    else:
        save_path = "plots/entropy_distribution.png"
        ensure_dir("plots")
        plt.savefig(save_path)
        plt.show()
    
    print(f"[Entropy] Saved entropy distribution plot to: {save_path}")
    return threshold


if __name__ == "__main__":
    """
    Example usage of the evaluation module.
    
    To use this module directly, uncomment and modify the example code below:
    
    import torch
    from torch.utils.data import DataLoader
    from torchvision import models
    import torch.nn as nn
    
    # Define your model and load weights
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('models/best_model.pth'))
    
    # Setup data loader
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define class names
    class_names = ['Cat', 'Dog', 'Bird']
    
    # Generate comprehensive evaluation
    criterion = nn.CrossEntropyLoss()
    avg_loss, accuracy, y_true, y_pred = classification_summary(
        model, test_loader, criterion, device, class_names
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, device, class_names=class_names)
    
    # Plot ROC curves
    plot_multiclass_roc(model, test_loader, device, 
                        num_classes=len(class_names),
                        class_names=class_names)
                        
    # Plot per-class metrics
    plot_per_class_metrics(y_true, y_pred, class_names)
    
    # Plot calibration curve
    plot_calibration_curve(model, test_loader, device, 
                          num_classes=len(class_names),
                          class_names=class_names)
    """
    
    print("This module provides model evaluation tools.")
import torch
import torch.nn as nn
from trainCE import LSTMClassifier  # Import your model class
from dataloader import create_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

def load_model(model, model_path):
    """Load saved model weights"""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def plot_roc_curves(labels, outputs):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for class_id in [1, 2, 3]:
        binary_labels = (labels == class_id).astype(int)
        class_probs = outputs[:, class_id-1]
        
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[class_id-1],
                label=f'Class {class_id} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.savefig('test_roc_curves.png')
    plt.close()

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    all_predicted = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted + 1  # Convert back to 1-3
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    
    # Convert to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_predicted = np.array(all_predicted)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    
    # Generate classification report
    class_report = classification_report(all_labels, all_predicted)
    
    # Save results
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Set Accuracy: {accuracy:.2f}%\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("Predicted\n")
        f.write("1\t2\t3\t<-- Actual\n")
        for i, row in enumerate(conf_matrix):
            row_str = "\t".join(map(str, row))
            f.write(f"{i+1} {row_str}\n")
        
        f.write("\nClassification Report:\n")
        f.write(class_report)
        
        # Save some example predictions
        f.write("\nExample Predictions:\n")
        f.write("Label\tPredicted\tProbabilities\n")
        for i in range(min(20, len(all_labels))):
            probs = all_outputs[i]
            f.write(f"{all_labels[i]:.0f}\t{all_predicted[i]:.0f}\t\t{probs}\n")
    
    # Plot ROC curves
    plot_roc_curves(all_labels, all_outputs)
    
    return accuracy, conf_matrix, class_report

def main():
    # Model parameters (should match training parameters)
    hidden_size = 256
    num_layers = 2
    dropout = 0.3
    
    # Initialize model
    model = LSTMClassifier(
        input_size=31,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load saved weights
    model = load_model(model, './best_model.pth')
    
    # Get test data loader
    _, _, test_loader = create_data_loaders(
        data_dir='./separated_wells_norm_removal',
        batch_size=32
    )
    
    # Evaluate model
    accuracy, conf_matrix, class_report = evaluate_model(model, test_loader)
    
    # Print results
    print(f"\nTest Set Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print("\nDetailed results have been saved to 'test_results.txt'")
    print("ROC curves have been saved to 'test_roc_curves.png'")

if __name__ == "__main__":
    main() 
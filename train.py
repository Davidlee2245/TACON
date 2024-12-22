import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataloader import create_data_loaders
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,      # Size of each input vector (31)
            hidden_size=hidden_size,     # Number of LSTM units
            num_layers=num_layers,       # Number of LSTM layers
            batch_first=True,           # Input shape: (batch, seq, feature)
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn1 = nn.BatchNorm1d(32) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)     # Output single value between 1-3
        
    def forward(self, x):
        # x shape: (batch_size, seq_length=7, input_size=31)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last time step output
        out = lstm_out[:, -1, :]        # Shape: (batch_size, hidden_size)
        
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Ensure output is between 1 and 3
        out = torch.clamp(out, min=1.0, max=3.0)
        
        return out

def get_predicted_class(output):
    """
    Convert regression output to class labels:
    1.0-1.5 -> 1
    1.5-2.5 -> 2
    2.5-3.0 -> 3
    """
    output = output.squeeze()
    predicted = torch.zeros_like(output)
    
    # Class 1: values < 1.5
    predicted = torch.where(output < 1.5, torch.tensor(1.0), predicted)
    # Class 2: 1.5 <= values < 2.5
    predicted = torch.where((output >= 1.5) & (output < 2.5), torch.tensor(2.0), predicted)
    # Class 3: values >= 2.5
    predicted = torch.where(output >= 2.5, torch.tensor(3.0), predicted)
    
    return predicted

def train_model(model, train_loader, val_loader, test_loader, num_epochs=150, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 ** (epoch // 50)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        """
            # Print some examples
            if epoch % 10 == 0:  # Print every 10 epochs
                print("\nExample predictions:")
                print("Output\tPredicted\tLabel")
                print("-" * 30)
                for i in range(min(5, len(outputs))):
                    pred = get_predicted_class(outputs[i])
                    print(f"{outputs[i].item():.2f}\t{pred.item():.0f}\t\t{labels[i].item():.0f}")
        """
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        all_predicted = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()
                
                predicted = get_predicted_class(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for statistics
                all_outputs.extend(outputs.squeeze().tolist())
                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

                # Print metrics
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'accuracy': accuracy
            }, 'best_model.pth')
            print(f'\nSaved new best model with validation loss: {val_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Update best validation accuracy
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            # Save best accuracy to txt file
            with open('best_validation_accuracy.txt', 'w') as f:
                f.write(f"Best Validation Accuracy: {best_val_accuracy:.2f}%\n")
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': best_val_accuracy
            }, 'best_model.pth')
            
    # After training, evaluate on test set
    evaluate_test_set(model, test_loader)

def evaluate_test_set(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    all_predicted = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = get_predicted_class(outputs)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for ROC curve
            all_outputs.extend(outputs.squeeze().tolist())
            all_labels.extend(labels.tolist())
            all_predicted.extend(predicted.tolist())
    
    # Calculate accuracy
    test_accuracy = 100 * correct / total
    print(f'\nTest Set Accuracy: {test_accuracy:.2f}%')
    
    # Save test results
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Set Accuracy: {test_accuracy:.2f}%\n")
        f.write("\nDetailed Results:\n")
        f.write("Label\tOutput\tPredicted\tCorrect?\n")
        f.write("-" * 45 + "\n")
        
        for i in range(len(all_labels)):
            is_correct = "✓" if all_labels[i] == all_predicted[i] else "✗"
            f.write(f"{all_labels[i]:.0f}\t{all_outputs[i]:.2f}\t{all_predicted[i]:.0f}\t\t{is_correct}\n")
    
    # Plot ROC curves
    plot_roc_curves(all_labels, all_outputs)

def plot_roc_curves(labels, outputs):
    # Convert to numpy arrays
    labels = np.array(labels)
    outputs = np.array(outputs)
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    # Plot ROC curve for each class
    for class_id in [1, 2, 3]:
        # Create binary labels for current class
        binary_labels = (labels == class_id).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, outputs)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=colors[class_id-1],
                label=f'ROC curve (class {class_id}) (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig('roc_curves.png')
    plt.close()

def main():
    # Modified hyperparameters
    batch_size = 32
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001  # Reduced learning rate
    num_epochs = 1000  # Increased epochs
    
    model = LSTMClassifier(
        input_size=31,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Add weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
    
    model.apply(init_weights)
    
    # Get data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='./separated_wells_norm',
        batch_size=batch_size
    )
    
    # Train model with test_loader
    train_model(model, train_loader, val_loader, test_loader, num_epochs, learning_rate)

if __name__ == "__main__":
    main()

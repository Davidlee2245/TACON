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
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 3)  # Changed to 3 outputs for 3 classes
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]
        
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

def train_model(model, train_loader, val_loader, test_loader, num_epochs=150, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
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
            # Convert labels to class indices (0-2)
            labels = labels.long() - 1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
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
                # Convert labels to class indices (0-2)
                labels_idx = labels.long() - 1
                val_loss += criterion(outputs, labels_idx).item()
                
                # Get predicted class
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted + 1  # Convert back to 1-3
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store probabilities for ROC curve
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Print metrics and save best model (rest of the code remains the same)
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
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
        
        scheduler.step()
        
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            with open('best_validation_accuracy.txt', 'w') as f:
                f.write(f"Best Validation Accuracy: {best_val_accuracy:.2f}%\n")
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")

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
        
        # Use the probability of the corresponding class
        class_probs = outputs[:, class_id-1]  # Use probabilities from softmax
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=colors[class_id-1],
                label=f'ROC curve (class {class_id}) (AUC = {roc_auc:.2f})')

def main():
    # Modified hyperparameters
    batch_size = 32
    hidden_size = 256
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
        data_dir='./separated_wells_norm_removal',
        batch_size=batch_size
    )
    
    # Train model with test_loader
    train_model(model, train_loader, val_loader, test_loader, num_epochs, learning_rate)

if __name__ == "__main__":
    main()

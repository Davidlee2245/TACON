import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataloader import create_data_loaders

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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)     # Output single value between 1-3
        
    def forward(self, x):
        # x shape: (batch_size, seq_length=7, input_size=31)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last time step output
        out = lstm_out[:, -1, :]        # Shape: (batch_size, hidden_size)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Ensure output is between 1 and 3
        out = torch.clamp(out, min=1.0, max=3.0)
        
        return out

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Print some training examples
        print("\nTraining Examples:")
        for i, (inputs, labels) in enumerate(train_loader):
            if i == 0:  # Print first batch examples
                outputs = model(inputs)
                predicted = torch.round(outputs.squeeze())
                
                # Print first 10 examples
                print("\nFirst 10 examples from training batch:")
                print("Label\tOutput\tPredicted")
                print("-" * 30)
                for j in range(min(10, len(labels))):
                    print(f"{labels[j]:.0f}\t{outputs[j].item():.2f}\t{predicted[j]:.0f}")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
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
                val_loss += criterion(outputs.squeeze(), labels.squeeze()).item()
                
                predicted = torch.round(outputs.squeeze())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels
                all_outputs.extend(outputs.squeeze().tolist())
                all_labels.extend(labels.tolist())
                all_predicted.extend(predicted.tolist())
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Print metrics
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Print validation examples
        print("\nValidation Examples:")
        print("Label\tOutput\tPredicted\tCorrect?")
        print("-" * 45)
        
        # Print first 10 validation examples
        for i in range(min(10, len(all_labels))):
            is_correct = "✓" if all_labels[i] == all_predicted[i] else "✗"
            print(f"{all_labels[i]:.0f}\t{all_outputs[i]:.2f}\t{all_predicted[i]:.0f}\t\t{is_correct}")
        
        # Print confusion matrix-like statistics
        print("\nPrediction Statistics:")
        for true_label in [1, 2, 3]:
            pred_counts = {1: 0, 2: 0, 3: 0}
            label_indices = [i for i, l in enumerate(all_labels) if l == true_label]
            for idx in label_indices:
                pred_counts[round(all_predicted[idx])] += 1
            total_count = sum(pred_counts.values())
            if total_count > 0:
                print(f"\nTrue Label {true_label}:")
                for pred_label, count in pred_counts.items():
                    percentage = (count / total_count) * 100
                    print(f"Predicted as {pred_label}: {count} ({percentage:.1f}%)")
        
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

def main():
    # Hyperparameters
    batch_size = 32
    hidden_size = 128  # Increased from 64
    num_layers = 3    # Increased from 2
    dropout = 0.3     # Increased from 0.2
    learning_rate = 0.001
    num_epochs = 100
    
    # Initialize model with modified architecture
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
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate)

if __name__ == "__main__":
    main()

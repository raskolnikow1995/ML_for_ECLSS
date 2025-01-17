import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, data_loader, num_epochs=30, learning_rate=0.001):
    """
    Trains an multi-variate multi-step LSTM model

    Args:
        model: The LSTM model instance
        data_loader: The DataLoader for training data
        num_epochs: The number of training epochs
        learning_rate: The learning rate for the optimizer

    Returns:
        model: The trained LSTM model.
    """
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs
            labels = labels.squeeze(-1)  # Shape: (batch_size, output_size)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Plotting the training loss
    plt.plot(range(1, num_epochs + 1), losses)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    return model


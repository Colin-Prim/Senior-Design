import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Simple3DLiftingModel(nn.Module):
    def __init__(self, input_dim=34, output_dim=51, hidden_dim=128):
        super(Simple3DLiftingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def load_data_2d_3d(data_2d_file, data_3d_file):
    # Load 2D and 3D data from numpy files
    data_2d = np.load(data_2d_file)
    data_3d = np.load(data_3d_file)

    # Flatten 2D keypoints and 3D keypoints for training
    data_2d = data_2d.reshape(data_2d.shape[0], -1)  # (N, J*2)
    data_3d = data_3d.reshape(data_3d.shape[0], -1)  # (N, J*3)

    return data_2d, data_3d

def train_model(train_loader, model, epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    return model

def main(data_2d_file, data_3d_file, output_model_file):
    # Load dataset
    data_2d, data_3d = load_data_2d_3d(data_2d_file, data_3d_file)

    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(data_2d, dtype=torch.float32),
                                  torch.tensor(data_3d, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model
    input_dim = data_2d.shape[1]  # Number of 2D input features
    output_dim = data_3d.shape[1]  # Number of 3D output features
    model = Simple3DLiftingModel(input_dim=input_dim, output_dim=output_dim)

    # Train the model
    model = train_model(train_loader, model, epochs=20)

    # Save the trained model
    torch.save(model.state_dict(), output_model_file)
    print(f"Model saved to {output_model_file}")

if __name__ == "__main__":
    main("2d_keypoints.npy", "3d_keypoints.npy", "3d_lifting_model.pth")

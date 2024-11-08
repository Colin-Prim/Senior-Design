import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the models directory to the Python path
sys.path.append(os.path.join(project_root, 'models'))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# Explicitly add the path to models and other dependencies
sys.path.append("/Users/jsm177y/Documents/GitHub/Senior Design/Senior-Design/models")
from models.pose_lifter_model import PoseLifterModel
from datasets.pose_dataset import PoseDataset
from configs.config import config
from utils import load_data, save_model

# Load data
train_data_2d = load_data(config["train_data_2d_path"])
train_data_3d = load_data(config["train_data_3d_path"])
val_data_2d = load_data(config["val_data_2d_path"])
val_data_3d = load_data(config["val_data_3d_path"])

# Prepare datasets and dataloaders
train_dataset = PoseDataset(train_data_2d, train_data_3d)
val_dataset = PoseDataset(val_data_2d, val_data_3d)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Model, criterion, optimizer, scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseLifterModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with gradient clipping and MAE tracking
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    total_mae = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate MAE for additional tracking
        total_mae += torch.sum(torch.abs(outputs - targets)).item()
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = total_mae / len(dataloader.dataset)
    return epoch_loss, epoch_mae

# Validation function with MAE tracking
def validate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            total_mae += torch.sum(torch.abs(outputs - targets)).item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = total_mae / len(dataloader.dataset)
    return epoch_loss, epoch_mae

# Main training loop with early stopping and learning rate adjustment
best_val_loss = float("inf")
early_stop_patience = 5
early_stop_counter = 0

for epoch in range(config["num_epochs"]):
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")
    
    train_loss, train_mae = train_model(model, train_loader, criterion, optimizer)
    print(f"Training Loss: {train_loss:.4f}, Training MAE: {train_mae:.4f}")
    
    val_loss, val_mae = validate_model(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")
    
    # Scheduler step based on validation loss
    scheduler.step(val_loss)
    
    # Check for best model and save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, config["model_save_path"])
        print("Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss for {early_stop_counter} epochs.")
    
    # Early stopping if no improvement for a set number of epochs
    if early_stop_counter >= early_stop_patience:
        print("Early stopping due to lack of improvement.")
        break

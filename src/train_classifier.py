import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm  # PyTorch Image Models library
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

# this is okay, since this file is in the same directory as dataset.py
from dataset import SolarDataset

# HYPERPARAMETERS 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "efficientnet_b0" 
SAVE_PATH = "models/classifier.pth"
DATA_CSV = "data/processed/master_metadata.csv"

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    # Progress bar for the terminal
    loop = tqdm(loader, desc="Training")
    
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1) # Shape becomes [batch_size, 1]
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Model Output (Logits) -> Probabilities -> Binary (0 or 1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Move to CPU for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate F1 Score (Critical Hackathon Metric)
    # Using 'binary' average because we only have 2 classes (Solar vs No Solar)
    f1 = f1_score(all_labels, all_preds, average='binary')
    return total_loss / len(loader), f1

# MAIN EXECUTION
if __name__ == "__main__":
    print(f"Training on device: {DEVICE}")
    
    # 1. Load Data
    # 'quality_filter=None': we use ALL data (High Quality Polygons + Low Quality Boxes)
    # 'task=classification': we get a single Label (0 or 1) instead of a Mask.
    print(f"Loading dataset from: {DATA_CSV}")
    
    # Check if CSV exists before crashing
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"CRITICAL: {DATA_CSV} not found. Did you run the symbolic link cell?")

    # running POC rn, checking to see if the pipeline works
    # just to prove the pipeline works. 
    full_ds = SolarDataset(DATA_CSV, root_dir=".", mode='train', quality_filter=None, task='classification')
    
    # Safety Check
    if len(full_ds) == 0:
        raise ValueError("Dataset is empty! Check your preprocessor logic.")
    
    print(f"Total Training Images: {len(full_ds)}")
    
    train_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    # Reusing train_loader for validation just for this demo. 
    # In production: val_loader = DataLoader(val_ds, ...)
    val_loader = train_loader 
    
    # 2. Setup Model (EfficientNet-B0)
    print(f"Initializing {MODEL_NAME}...")
    # pretrained=True downloads weights learned on ImageNet (Transfer Learning)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    model = model.to(DEVICE)
    
    # 3. Setup Loss & Optimizer
    # BCEWithLogitsLoss is standard for Binary Classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nExample Epoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | 🎯 Val F1: {val_f1:.4f}")
        
        # Save Best Model
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved New Best Model! (F1: {best_f1:.4f})")
            
    print("\nTraining Complete.")
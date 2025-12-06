import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import SolarDataset

# HYPERPARAMETERS
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "resnet50"
WEIGHTS = "imagenet"
SAVE_PATH = "models/segmenter.pth"

os.makedirs("models", exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training Segmenter")
    
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    train_ds = SolarDataset(
        "data/processed/master_metadata.csv", 
        root_dir=".", 
        mode='train', 
        quality_filter='high', 
        task='segmentation'
    )
    
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 2. Setup Model
    print(f"Loading U-Net++ with {ENCODER} backbone...")
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=WEIGHTS, 
        in_channels=3, 
        classes=1, 
        activation=None 
    )
    model = model.to(DEVICE)
    
    # 3. Loss & Optimizer
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    
    def criterion(pred, target):
        return dice_loss(pred, target) + bce_loss(pred, target)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Train
    min_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch(model, loader, criterion, optimizer)
        print(f"Loss: {loss:.4f}")
        
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model Saved! (Loss: {min_loss:.4f})")
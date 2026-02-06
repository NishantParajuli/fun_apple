
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .mapper_model import BlindPainterMapper
import os
import random
from tqdm import tqdm

class EmbeddingDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.audio_emb = data["audio_embeddings"].float()
        self.image_emb = data["image_embeddings"].float()
        
    def __len__(self):
        return len(self.audio_emb)
    
    def __getitem__(self, idx):
        return self.audio_emb[idx], self.image_emb[idx]

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    dataset_path = "models/embeddings.pt"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    dataset = EmbeddingDataset(dataset_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = BlindPainterMapper().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion = nn.CosineEmbeddingLoss()
    
    epochs = 100
    best_val_loss = float('inf')
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for audio_emb, img_emb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            audio_emb, img_emb = audio_emb.to(device), img_emb.to(device)
            
            optimizer.zero_grad()
            pred_emb = model(audio_emb)
            
            target = torch.ones(audio_emb.size(0)).to(device)
            loss = criterion(pred_emb, img_emb, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audio_emb, img_emb in val_loader:
                audio_emb, img_emb = audio_emb.to(device), img_emb.to(device)
                pred_emb = model(audio_emb)
                target = torch.ones(audio_emb.size(0)).to(device)
                loss = criterion(pred_emb, img_emb, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "models/mapper_best.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()

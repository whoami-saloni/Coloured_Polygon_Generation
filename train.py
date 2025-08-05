import os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
from Coloured_Polygon_Generation.Preprocess import PolygonColorDataset
from Coloured_Polygon_Generation.model import UNet
wandb.init(project="colored-polygon-unet")

    # Step 1: Load color names from both training and validation JSONs
def extract_colors(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return set(entry['colour'] for entry in data)

def train():

    train_colors = extract_colors("/Users/salonisahal/Col_Polygon_Gen/Data/training/data.json")
    val_colors = extract_colors("/Users/salonisahal/Col_Polygon_Gen/Data/validation/data.json")
    all_colors = sorted(train_colors.union(val_colors))

    # Step 2: Build color2idx mapping dynamically
    color2idx = {color: idx for idx, color in enumerate(all_colors)}
    print("Detected colors:", color2idx)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Datasets
    train_ds = PolygonColorDataset(
        "/Users/salonisahal/Col_Polygon_Gen/Data/training/inputs",
        "/Users/salonisahal/Col_Polygon_Gen/Data/training/outputs",
        "/Users/salonisahal/Col_Polygon_Gen/Data/training/data.json",
        color2idx, transform
    )
    val_ds = PolygonColorDataset(
        "/Users/salonisahal/Col_Polygon_Gen/Data/validation/inputs",
        "/Users/salonisahal/Col_Polygon_Gen/Data/validation/outputs",
        "/Users/salonisahal/Col_Polygon_Gen/Data/validation/data.json",
        color2idx, transform
    )

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, optimizer, loss
    model = UNet(color2idx=color2idx,color_embedding_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        for (img, color), target in train_loader:
            img, color, target = img.to(device), color.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img, color)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        wandb.log({"train_loss": total_loss / len(train_loader), "epoch": epoch})

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for (img, color), target in val_loader:
                img, color, target = img.to(device), color.to(device), target.to(device)
                out = model(img, color)
                val_loss += criterion(out, target).item()
            wandb.log({"val_loss": val_loss / len(val_loader), "epoch": epoch})

    torch.save(model.state_dict(), "/Users/salonisahal/Col_Polygon_Gen/Models/colored_polygon_unet.pth")

if __name__ == "__main__":
    train()
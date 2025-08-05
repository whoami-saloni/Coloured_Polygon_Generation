import os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, color_embedding_dim, color2idx):
        super().__init__()
        self.embed_color = nn.Linear(len(color2idx), color_embedding_dim)
        self.encoder1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128 + color_embedding_dim, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, img, color):
        e1 = self.encoder1(img)
        e2 = self.encoder2(self.pool1(e1))

        color_emb = self.embed_color(color).unsqueeze(2).unsqueeze(3)  # shape: (B, E, 1, 1)
        color_emb = color_emb.expand(-1, -1, e2.shape[2] // 2, e2.shape[3] // 2)
        bottleneck_input = torch.cat([self.pool2(e2), color_emb], dim=1)

        b = self.bottleneck(bottleneck_input)
        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))
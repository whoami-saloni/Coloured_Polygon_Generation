import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class PolygonColorDataset(Dataset):
    def __init__(self, input_dir, output_dir, mapping_file, color2idx, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.color2idx = color2idx
        self.transform = transform

        with open(mapping_file, 'r') as f:
            self.data_map = json.load(f)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
      entry = self.data_map[idx]
      input_path = os.path.join(self.input_dir, entry['input_polygon'])
      output_path = os.path.join(self.output_dir, entry['output_image'])
      color_name = entry['colour']
      color_idx = self.color2idx[color_name]

      img_in = Image.open(input_path).convert('L')  # grayscale
      img_out = Image.open(output_path).convert('RGB')

      if self.transform:
        img_in = self.transform(img_in)
        img_out = self.transform(img_out)

      color_tensor = F.one_hot(torch.tensor(color_idx), num_classes=len(self.color2idx)).float()
      return (img_in, color_tensor), img_out
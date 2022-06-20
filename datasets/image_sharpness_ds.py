from torch.utils.data import Dataset
import matplotlib.pyplot as plt


import os
import pandas as pd
import numpy as np
from PIL import Image
import torch as th


class ImageSharpnessDS(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path)
        img = (np.asarray(img, dtype=np.float32) - 2**15) / 2**16
        img = np.array(img, dtype=np.float32)
        img = th.from_numpy(img)

        img = th.unsqueeze(img, dim=0)
        if self.transform:
            img = self.transform(img)

        label = self.img_labels.iloc[idx, 1]
        name = self.img_labels.iloc[idx, 0]
        label = th.tensor(label, dtype=th.float32)
        label = th.unsqueeze(label, dim=0)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label, name

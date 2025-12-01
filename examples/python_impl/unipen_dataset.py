# Copyright (C) 2025 Piotr Jabłoński
# Extended copyright information can be found in the LICENSE file.

import os

import cv2
from torch.utils.data import Dataset


class UnipenDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "trn", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        self.data = []
        self.targets = []
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                self.data.append(image_path)
                self.targets.append(int(label))  # numeric value of ASCII char

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)
        return img, label

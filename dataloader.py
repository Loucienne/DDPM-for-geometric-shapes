import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader


class train_dataloader(Dataset):
    def __init__(self, image_dir, annotation_path, objects_path, transform=None):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Load the annotations (filename to list of objects)
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        # Load the object classes
        with open(objects_path, 'r') as f:
            self.object_list = json.load(f)

        self.obj_to_idx = {obj: idx for idx, obj in enumerate(self.object_list)}
        self.image_filenames = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        # Multi-label: convert object names to a multi-hot vector
        labels = self.annotations[image_filename]
        target = torch.zeros(len(self.object_list), dtype=torch.float32)
        for label in labels:
            if label in self.obj_to_idx:
                target[self.obj_to_idx[label]] = 1.0

        return image, target




class test_dataloader(Dataset):
    def __init__(self, annotation_path, objects_path, transform=None):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Load the annotations (filename to list of objects)
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        # Load the object classes
        with open(objects_path, 'r') as f:
            self.object_list = json.load(f)

        self.obj_to_idx = {obj: idx for idx, obj in enumerate(self.object_list)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Multi-label: convert object names to a multi-hot vector
        labels = self.annotations[idx]
        target = torch.zeros(len(self.object_list), dtype=torch.float32)
        for label in labels:
            if label in self.obj_to_idx:
                target[self.obj_to_idx[label]] = 1.0
        return target, labels

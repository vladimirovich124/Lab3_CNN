import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)

label_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        self.image_files = list(self.labels.keys())
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels.values())))}
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[self.image_files[idx]]
        label = self.label_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(os.path.join(config.data_dir, 'batches'), os.path.join(config.data_dir, 'train_labels.json'), transform=transform)
    val_dataset = CustomDataset(os.path.join(config.data_dir, 'batches'), os.path.join(config.data_dir, 'val_labels.json'), transform=transform)
    test_dataset = CustomDataset(os.path.join(config.data_dir, 'test'), os.path.join(config.data_dir, 'test_labels.json'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    images = []
    labels = []
    for image, label in batch:
        if image is not None and label is not None:
            if label in label_to_idx:
                images.append(image)
                labels.append(label_to_idx[label])
            else:
                logger.warning(f"Skipping image with unknown label: {label}")

    if not images:
        return None, None  

    images = torch.stack(images)

    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels


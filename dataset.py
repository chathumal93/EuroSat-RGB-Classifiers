from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
import torch

# Transformations
transforms = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.3445, 0.3803, 0.4077], [0.0915, 0.0652, 0.0553])
])


class EuroSat(Dataset):
    def __init__(self, data, transform=transforms, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['label']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


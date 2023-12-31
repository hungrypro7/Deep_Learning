import os
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, root='/content/drive/MyDrive/UTKFace/data/part1/', split='train', transform=None):
        self.transform = transform
        self.root = root
        self.images = []

        for filename in os.listdir(root):
            if filename.endswith('.jpg'):
                self.images.append(filename)

        # Split data into train, validation, test sets (0.8 : 0.15 : 0.15)
        train_data, test_data = train_test_split(self.images, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

        if split == 'train':
            self.images = train_data
        elif split == 'val':
            self.images = val_data
        elif split == 'test':
            self.images = test_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        labels = []
        # Read images
        img_path = os.path.join(self.root, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images = self.transform(img)

        # Extract labels from the filename
        parts = self.images[idx].split('_')
        age, gender, race, _ = parts
        age = int(age)
        gender = int(gender)
        race = int(race)
        labels = torch.tensor([age, gender, race], dtype=torch.float32)
        print(images, labels)
        return images, labels


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

image_transform = transforms.Compose([
                    transforms.ToTensor(),     # 이미지를 텐서로 변환
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    transforms.Resize((224, 224))
                  ])

train_set = CustomDataset(split='train', transform=image_transform)
val_set = CustomDataset(split='val', transform=image_transform)
test_set = CustomDataset(split='test', transform=image_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

for x, y in train_loader:
    print(x.shape)
    print(y.shape)
    break
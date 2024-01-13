import os
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
import math
import random

# Configure device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CustomDataset for UTKFace
class CustomDataset(Dataset):
    def __init__(self, root='/content/drive/MyDrive/UTKFace/data/new_cropped_faces/', split='train', transform=None):
        self.transform = transform
        self.root = root
        self.images = []

        for filename in os.listdir(root):
            if filename.endswith('.jpg'):
                self.images.append(filename)

        # Split data into train, validation, test sets (0.7 : 0.15 : 0.15)
        train_data, test_data = train_test_split(self.images, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

        # 각 age에 대한 데이터 개수 파악
        age_counts = {}
        for filename in self.images:
            age = int(filename.split('_')[0])
            age_counts[age] = age_counts.get(age, 0) + 1

        # total : 24101
        if split == 'train':
            self.images = train_data    # 16871
        elif split == 'val':
            self.images = val_data    # 3615
        elif split == 'test':
            self.images = test_data   # 3616

        # 각 age에 대한 데이터 개수의 역수를 계산하여 확률로 사용
        total_data = len(self.images)
        self.probabilities = [1 / age_counts[int(filename.split('_')[0])] for filename in self.images]
        self.probabilities = [prob / sum(self.probabilities) for prob in self.probabilities]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        labels = []

        # 확률에 따라 데이터 샘플링
        selected_index = random.choices(range(len(self.images)), weights=self.probabilities)[0]

        # Read images
        img_path = os.path.join(self.root, self.images[selected_index])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images = self.transform(img)

        # Extract labels from the filename
        parts = self.images[selected_index].split('_')
        age, gender, race, _ = parts
        age = int(age)
        gender = int(gender)
        race = int(race)
        labels = torch.tensor([age, gender, race], dtype=torch.float32)
        return images, labels

image_transform = transforms.Compose([
                    transforms.ToTensor(),     # 이미지를 텐서로 변환
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # 이미지 정규화 수행
                    transforms.Resize((224, 224))     # 이미지 크기 고정
                  ])

train_set = CustomDataset(split='train', transform=image_transform)
val_set = CustomDataset(split='val', transform=image_transform)
test_set = CustomDataset(split='test', transform=image_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

for x, y in train_loader:
    print(x.shape)    # images
    print(y.shape)    # label
    break


class RegressionNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(RegressionNN, self).__init__()
        # Use a pre-trained ResNet18 as a feature extractor
        self.resnet = models.resnet152(pretrained=True)

        # Modify the last fully connected layer for regression (1 output neuron for age prediction)
        # Add dropout to the fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, 1)
        )

    def forward(self, x):
        return self.resnet(x)

# Instantiate the regression model
regression_model = RegressionNN(dropout_rate=0.5).to(DEVICE)
# Loss function for regression (Mean Squared Error) 회
age_loss = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.0001)

def evaluate(model, test_loader):
    # Evaluation loop for regression
    model.to(DEVICE)
    model.eval()
    total_mae = 0.0
    age_acc = 0

    with torch.no_grad():
        for X, y in test_loader:
            # Extract age labels (index 0 in the label tensor)
            age_labels = y[:, 0].view(-1, 1)

            # Move data to the device
            X, age_labels = X.to(DEVICE), age_labels.to(DEVICE)

            # Forward pass
            predictions = regression_model(X)

            # age_acc += (predictions.round() == age_labels).sum().item()

            mae = nn.L1Loss()(predictions, age_labels)
            total_mae += mae.item()

    # age_acc = age_acc / len(test_loader)
    # print(f"Age accuracy: {age_acc*100}%")

    return total_mae, total_mae / len(test_loader)

# Training loop for regression
num_epochs = 20  # You can adjust the number of epochs
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)    # 진행 상황을 시각적으로 표시하는 라이브러리
    total_loss = 0.0

    for _, (X, y) in loop:
        # Extract age labels (index 0 in the label tensor)
        age_labels = y[:, 0].view(-1, 1)

        # Move data to the device
        X, age_labels = X.to(DEVICE), age_labels.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = regression_model(X)

        # Compute the loss
        loss = age_loss(predictions, age_labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update the total loss for this epoch
        total_loss += loss.item()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=total_loss / len(train_loader))

    # Validation
    validation_total_mse, validation_average_mse = evaluate(regression_model, val_loader)

    # Print validation results
    print(f"Validation - Epoch [{epoch+1}/{num_epochs}], Average Mean Absolute Error: {validation_average_mse}")


# Evaluate the model on the test set
total_mse, average_mse = evaluate(regression_model, test_loader)

# Calculate the average Mean Squared Error on the test set
print(f"Average Mean Absolute Error on the test set: {average_mse}")

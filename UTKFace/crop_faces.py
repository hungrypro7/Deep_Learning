# 폴더에 들어있는 인물 사진들 중 얼굴 부분만 잘라내어 새로운 폴더에 따로 저장하는 코드입니다.
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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from google.colab.patches import cv2_imshow

# Configure device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root='/content/drive/MyDrive/UTKFace/data/part1/'

images = []
for filename in os.listdir(root):
    if filename.endswith('.jpg'):
        images.append(filename)

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image using Haar Cascade
def detect_faces_haar(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Convert the (x, y, w, h) values to the format expected by the code
    boxes = [[x, y, x+w, y+h] for (x, y, w, h) in faces]

    return boxes

new_output_dir = '/content/drive/MyDrive/UTKFace/data/new_cropped_faces/'
os.makedirs(new_output_dir, exist_ok=True)

for i in range(len(images)):
    img_path = os.path.join(root, images[i])
    boxes = detect_faces_haar(img_path)
    # Crop the image to the detected face
    if len(boxes) > 0:
        img = cv2.imread(img_path)

        # Assuming there is only one face, process the first detected face
        box = boxes[0]

        # Display the image with bounding box
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Crop and display the detected face
        face = img[box[1]:box[3], box[0]:box[2]]
        # cv2_imshow(face)

        new_output_path = os.path.join(new_output_dir, images[i])  # Keep the same filename
        cv2.imwrite(new_output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
import torch
from torch.utils.data import Dataset, RandomSampler
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image, ImageMath
import random
import math
import numpy as np

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
#import wandb

class MultiMNIST(Dataset):
    def __init__(self, root='./data', train=True, rot_mode=True, size_mode=True, noise_mode=True,
                 transform=None, target_transform=None, bb_transform=None, download=True):
        self.transform = transform          # 데이터 변환을 수행하는 함수
        self.target_transform = target_transform    # 타겟 데이터에 대한 변환을 수행하는 함수
        self.bb_transform = bb_transform
        self.digit_length = 2       # 합성할 이미지 숫자 개수
        self.mnist_loader = MNIST(root=root, train=train, download=download)
        self.mnist_length = math.floor(len(self.mnist_loader) / self.digit_length)
        self.train = train          # 클래스의 학습 모드 여부
        self.rot_mode = rot_mode    # 회전 모드 사용 여부
        self.size_mode = size_mode          # 크기 모드 사용 여부
        self.noise_mode = noise_mode        # 노이즈 모드 사용 여부

    def __getitem__(self, item):        # 하나의 아이템을 가져옴
        if not self.train:      # train이 False 일 때 (test 모드), 항상 동일한 무작위성을 보장하기 위해 사용됨
            self.random = random.Random(item)
            state = np.random.get_state()
            np.random.seed(item)
            self.np_local_state = np.random.get_state()
            np.random.set_state(state)
        else:
            self.random = random
        # images, labels 리스트 초기화
        images = []
        labels = []
        for i in range(self.digit_length):
            if self.train:
                image, label = self.mnist_loader[random.randrange(len(self.mnist_loader))]  # MNIST 데이터셋의 길이 중에서 무작위로 인덱스를 선택해 해당하는 Image, label을 가져옴
            else:   # 테스트 모드일 때
                image, label = self.mnist_loader[item * self.digit_length + i]  # 데이터가 무작위로 선택되지 않고, 선택된 인덱스에 해당하는 Image, label을 가져옴
            images.append(image)
            labels.append(torch.tensor(label))

        # 이 값들은 이미지 합성에 사용됨
        rot_range = 10.0 if self.rot_mode else 0.0
        size_range = (20, 32) if self.size_mode else (28, 28)
        noise = (0.05, 0.0) if self.noise_mode else (0.0, 0.0)

        images, bb1, bb2, label_reverse = self.combine(*images, rot_range=rot_range, size_range=size_range, noise=noise)    # 이미지를 합성하고 bounding box 정보를 얻음
        bbs = [bb1, bb2]
        if label_reverse is True:   # label_reverse 값에 따라 bounding box를 조정함
            labels.reverse()
            bbs.reverse()
        labels = torch.stack(labels)
        bbs = torch.tensor(bbs).t()

        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.bb_transform:
            bbs = self.bb_transform(bbs)

        return images, labels, bbs

    def __len__(self):                  # 데이터 셋의 전체 길이를 반환함
        return self.mnist_length

    def combine(self, im1, im2, rot_range=0.0, size_range=(28, 28), noise=(0.0, 0.0)):      # 두 이미지를 합성하여 새로운 이미지와 bounding box 정보를 반환하는 메서드
        stagesize = [64, 64]    # 이미지 합성에 사용될 변수들
        imsize = [48, 48]
        resample = Image.BILINEAR

        if rot_range > 0.0:         # 이미지 회전이 활성화되어 있다면, 두 이미지를 무작위로 회전시킴
            rot1 = self.random.uniform(-rot_range, rot_range)   # 회전 각도 지정
            rot2 = self.random.uniform(-rot_range, rot_range)
            im1 = im1.rotate(rot1, resample=resample, expand=True, fillcolor=0)     # 해당 각도만큼 회전 시킴
            im2 = im2.rotate(rot2, resample=resample, expand=True, fillcolor=0)

        size1 = self.random.randrange(size_range[0], size_range[1] + 1)  # 첫번째 이미지 크기를 size_range 범위 내에서 무작위로 선택함
        size2 = self.random.randrange(size_range[0], size_range[1] + 1)
        im1size = im1.size          # 이미지 크기 정보를 저장
        im2size = im2.size
        maxim1size = max(im1size)       # 첫번쨰 이미지의 가로 또는 세로 최대 크기를 저장
        if maxim1size != size1:     # 이미지 크기 비율을 계산해 이미지를 크기에 맞게 조정함
            ratio1 = size1 / maxim1size
            im1 = im1.resize((round(im1size[0] * ratio1), round(im1size[1] * ratio1)), resample=resample)
            im1size = im1.size
        maxim2size = max(im2size)       # 두 번째 이미지도 동일
        if maxim2size != size2:
            ratio2 = size2 / maxim2size
            im2 = im2.resize((round(im2size[0] * ratio2), round(im2size[1] * ratio2)), resample=resample)
            im2size = im2.size

        im1full = Image.new(im1.mode, imsize)       # 첫 번째 이미지를 채울 새로운 이미지를 생성함, imsize와 같은 크기를 가지며, 모든 픽셀값은 0으로 초기화됨
        im1off = (self.random.randrange(imsize[0] - im1size[0]), self.random.randrange(imsize[1] - im1size[1]))     # im1full을 삽입할 위치를 무작위로 선택함
        im1full.paste(im1, im1off)      # 무작위로 선택된 위치에 첫 번째 이미지 im1을 삽입함
        im2full = Image.new(im2.mode, imsize)
        im2off = (self.random.randrange(imsize[0] - im2size[0]), self.random.randrange(imsize[1] - im2size[1]))
        im2full.paste(im2, im2off)

        label_reverse = im1off[0] > im2off[0]       # 각 이미지가 삽입될 위치를 나타냄

        im = ImageMath.eval('max(a, b)', a=im1full, b=im2full)      # 픽셀별로 더 큰 값을 가지는 픽셀로 이루어진 새로운 이미지 im을 생성합니다. 이를 통해 두 이미지를 결합

        stage = Image.new(im.mode, stagesize)       # 새로운 이미지를 생성함
        imoff = (self.random.randrange(stagesize[0] - imsize[0]), self.random.randrange(stagesize[1] - imsize[1]))      # im이 stage에 삽입될 위치를 나타냄
        stage.paste(im, imoff)

        # 이미지에 노이지를 추가하고 해당 이미지에 대한 bounding box를 계산하는 단계
        if self.train:
            im_noise = np.random.rand(stagesize[1], stagesize[0])       # stagesize 크기의 무작위 노이즈를 생성함
        else:
            state = np.random.get_state()
            np.random.set_state(self.np_local_state)
            im_noise = np.random.rand(stagesize[1], stagesize[0])
            self.np_local_state = np.random.get_state()
            np.random.set_state(state)
        # im_noise = np.array([[self.random.random() for i in range(imsize[0])] for j in range(imsize[1])])
        # 소금-후추 노이즈를 적용함
        salt = Image.fromarray(((im_noise < noise[0]) * 255).astype('uint8')).convert(stage.mode)
        pepper = Image.fromarray(((im_noise > 1. - noise[1]) * 255).astype('uint8')).convert(stage.mode)
        stage = ImageMath.eval('min(a + b, 255)', a=stage, b=salt)
        stage = ImageMath.eval('max(a - b, 0)', a=stage, b=pepper)
        stage = stage.convert('L')
        # 첫번째 및 두번째 이미지에 대한 bounding box의 좌표를 나타냄
        bb1 = (imoff[1] + im1off[1],
               imoff[1] + im1off[1] + im1size[1],
               imoff[0] + im1off[0],
               imoff[0] + im1off[0] + im1size[0])
        bb2 = (imoff[1] + im2off[1],
               imoff[1] + im2off[1] + im2size[1],
               imoff[0] + im2off[0],
               imoff[0] + im2off[0] + im2size[0])

        return stage, bb1, bb2, label_reverse

# Residual block
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()         # Resnet Shortcut (기존의 값과 CNN+BN 한 결과를 더함)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):       # forward propagation
        out = F.relu(self.bn1(self.conv1(x)))   # conv -> bn1 -> relu -> conv2 -> bn2
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)     # skip connection
        out = F.relu(out)
        return out

# Resnet 모델
class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # input block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # residual blocks
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)  # Fully Connected Layer
        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

#if __name__ == '__main__':      # MultiMNIST 데이터 셋을 생성하고, 데이터를 로드하여 시각화하는 부분
# MultiMNIST 데이터 셋을 생성하고 각 샘플은 두 개의 MNIST 숫자 이미지를 합성함
image_transform = transforms.ToTensor()

dataset = MultiMNIST(transform=image_transform)

a, b, c = dataset[0]

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

iterator = iter(data_loader)
image, label, bb = next(iterator)

cnt = 0
for image, label, bb in data_loader:
    cnt += 1
    interval = 1
    image_ = image[0]
    bb_ = bb[0]
    label_ = label[0]

    #wandb.init(project='multimnist')
    #wandb.run.name = 'resnet_01'
    #wandb.run.save()

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)


batch_size = 128        # 데이터 개수
EPOCHS = 300            # 학습 횟수

image_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),     # 이미지를 텐서로 변환
                                      transforms.Normalize((0.5), (0.5))])

# MNIST 데이터 갖고 오기
train_dataset = MultiMNIST(train=True, rot_mode=True, size_mode=True, noise_mode=True, transform=image_transform)
test_dataset = MultiMNIST(train=False, rot_mode=True, size_mode=True, noise_mode=True, transform=image_transform)
# 다운로드한 MNIST 데이터셋을 batch_size 단위로 분리해 저장
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Initialize
model = ResNet().to(DEVICE)
model.apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)   # 학습률 : 모델의 가중치를 0.0001만큼 조정
criterion = nn.CrossEntropyLoss()    # 다중 분류를 위한 손실 함수
print(model)

# 이미지 데이터와 레이블 데이터를 이용해 MLP 모델을 학습
def train(model, train_loader, optimizer):
    model.train()       # MLP 모델을 학습 상태로 지정
    total_loss = 0.
    n = 0

    for batch_idx, (image, label, bb) in enumerate(train_loader): # 기존에 정의한 GPU에 데이터를 할당
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        label_temp = torch.zeros(label.shape[0], 1).to(DEVICE)
        #label_temp = torch.zeros(label.shape[0], 1)
        label_temp = label[:, 0] * 10 + label[:, 1]

        optimizer.zero_grad()       # optimizer Gradient 초기화
        # Forward
        output = model(image)       # Input : 이미지 데이터 / Output 계산
        # Backward
        loss = criterion(output, label_temp)     # loss 값 계산
        loss.backward()     # Back 통해 계산된 Gradient 값을 각 파라미터에 할당함
        optimizer.step()    # 각 파라미터에 할당된 Gradient 값을 이용해 파라미터 값을 업데이트함
        total_loss += loss.item()   # Propagation을 각 batch에 대한 손실값을 더해줌 (scalar)

        n += output.shape[0]    # 각 batch에 포함된 샘플 수를 더해줌

    print(f"Train Epoch: {epoch}, Train Loss: {total_loss / n:.6f}")
    '''args = {
        "Train Loss": total_loss / n,
        "epoch": epoch
    }
    wandb.log(args)'''

# 검증 데이터에 대한 모델 성능을 확인
def evaluate(model, test_loader):
    model.eval()    # MLP 모델을 평가 상태로 지정
    test_loss = 0
    correct = 0
    n = 0

    with torch.no_grad():       # 파라미터 업데이트 방지
        for image, label, bb in test_loader:      # image, label, bounding_box
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            label_temp = torch.zeros(label.shape[0], 1).to(DEVICE)
            #label_temp = torch.zeros(label.shape[0], 1)
            label_temp = label[:, 0] * 10 + label[:, 1]

            output = model(image)
            test_loss += criterion(output, label_temp).item()    # loss 값 계산
            prediction = output.max(1, keepdim = True)[1]    # 계산된 벡터값 내 가장 큰 값인 위치에 대해 해당 위치에 대응하는 클래스로 예측
            correct += prediction.eq(label_temp.view_as(prediction)).sum().item()   # 예측한 클래스 값과 실제 레이블이 의미하는 클래스가 맞으면 correct에 더해 올바르게 예측한 횟수를 저장
    test_loss /= len(test_loader.dataset)        # 평균 loss 값
    test_accuracy = 100. * correct / len(test_loader.dataset)   # rest_loader 데이터 중 얼마나 맞췄는지를 계산해 정확도를 계산함
    '''wandb.log({
        "Test loss" : test_loss,
        "Test accuracy" : test_accuracy
    })'''
    return test_loss, test_accuracy


for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f"\n[EPOCH: {epoch}], \tTest Loss: {test_loss:.4f}, \tTest Accuracy: {test_accuracy:.2f} %\n")
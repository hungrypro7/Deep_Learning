Term Project Report
이찬영 (2019098068, 미디어테크놀로지전공)
본 프로젝트는 입력된 이미지를 dog, cat, horse 등의 10가지 동물 유형으로 분류하는 신경망을
학습시켜서 일반적인 상황에서 최고의 성능을 내게 하는 프로젝트입니다. 원래 있는 skeleton 
code에 hyper parameter tuning, data augmentation 등의 기능을 추가하여 모델의 성능을 끌어 올
리는 것이 최종 목표입니다. 본 프로젝트는 한양대학교 ERICA 컴퓨터학부 3학년 2학기 딥러닝
(정우환 교수님) 수업의 일환으로 수행되었습니다.
1. 수정한 파일 설명
_utils.py : 해당 파일은 PyTorch를 사용하여 이미지 classification 모델을 학습하기 위한 데이터 로
더를 생성합니다. args에 있는 데이터 경로에서 ImageFolder 데이터셋을 생성하고, 학습/테스트
데이터로 분할되어 있는 것에, 검증용 데이터 분할을 추가해서 각각에 대한 데이터 로더를 생성
하였습니다. 학습, 검증, 테스트 데이터 비율은 0.8:0.1:0.1로 설정하였습니다.
train.py : PyTorch가 제공하는 여러가지 모델들을 불러와서 테스트를 진행하였으며, 최종 모델로
resnet18을, 가중치로는 ResNet50_Weights를 선택했습니다. (2번 참고) 손실 함수는 CrossEntropy
를 사용하였으며, 옵티마이저 함수는 Adam을 사용하였습니다. 또한, _utils.py에 추가해주었던 검
증 데이터에 대한 검증 기능을 추가하였으며 각 epoch마다 손실과 정확도를 출력해 이를 통해
모델의 성능을 평가하도록 수정하였습니다.
2. 테스트 설명
Colab을 통해 여러가지 딥러닝 classification 모델을 적용해보고 테스트 해보았습니다. 다음은 테
스트를 수행한 결과입니다.
(1) Resnet18
Model Resnet18 Resnet18 Resnet18 Resnet18
Weight ResNet18_Weights ResNet18_Weights ResNet18_Weights.IMAGENET1K_V1 ResNet18_Weights
Epoch 10 15 15 5
Learning rate 0.001 0.001 0.001 0.00001
Test Accuracy 0.89037 0.90794 0.87820 0.97174
Result.txt 58/100 62/100 68/100 70/100
Resnet18 모델을 적용해 보았으며, 처음에는 epoch과 weight를 바꿔 보았으나, 한계가 존재했으
며, learning rate와 epoch을 더 줄였더니 더 성능이 개선된 것을 확인할 수 있었습니다.
(2) resnet50
Model Resnet50 Resnet50 Resnet50 Resnet50
Weight
ResNet50_Weights.IMA
GENET1K_V2
ResNet50_Weights.DEF
AULT
ResNet50_Weights.IMA
GENET1K_V1
ResNet50_Weights.IMA
GENET1K_V1
Epoch 15 5 5 5
Learning rate 0.001 0.0001 0.0001 0.00001
Test Accuracy 0.92822 0.97556 0.91333 0.97633
Result.txt 64/100 68/100 64/100 71/100
Resnet50 모델을 적용해 보았으며, 확실히 Resnet18에 비해 성능이 좋다는 것을 확인할 수 있었
고, weight를 바꿔보았더니 성능이 더 개선된 것을 확인할 수 있었습니다.
(3) 그 밖의 다른 모델들
Model googleNet Resnet101 Resnet152 Resnet152
Weight
GoogLeNet_Weights.IM
AGENET1K_V1
ResNet101_Weights.DE
FAULT
ResNet152_Weights.DE
FAULT
ResNet152_Weights.IM
AGENET1K_V2
Epoch 5 5 10 5
Learning rate 0.0001 0.001 0.001 0.001
Test Accuracy 0.91180 0.90874 0.88774 0.90951
Result.txt 65/100 63/100 63/100 63/100
Resnet18, Resnet50을 제외한 다른 모델들을 갖고 와서 적용해 보았으며, 성능이 생각보다 잘 나
오지 않았습니다. 또한 용량 문제로 인해 현실적으로 적용할 수 없는 한계도 있었습니다. 시도를
통해 꼭 모델 숫자가 더 높다고 해서 무조건 성능이 잘 나오지는 않다는 것을 알게 되었습니다.
여러 번의 테스트를 수행한 결과, 빨간색 글자 부분이 성능이 제일 좋았던 것으로 확인되었고, 용
량문제 때문에 resnet18을 최종 적용하였습니다. 마지막으로, Result.txt에 있는 100개 부문 맞은
개수를 따져보니 다음과 같이 나오는 것을 확인할 수 있었습니다.
3. 과제를 수행하면서 느낀 점
이번 과제를 수행하며 Image classification을 위한 딥러닝 모델을 개선하면서 learning rate, batch 
size, epoch 등의 하이퍼파라미터를 효과적으로 조정하는 것이 모델의 성능에 큰 영향을 미칠 수
있다는 것을 경험할 수 있었습니다. 또한 100% 정확한 모델은 없으며 수많은 실험을 통해 최적
의 하이퍼파라미터를 찾는 것이 중요하다는 것도 알게 되었습니다. 뿐만 아니라, 데이터의 품질과
양, 그리고 적절한 전처리가 모델의 학습에 큰 영향을 미치며, 데이터셋을 잘 이해하고 적절한 변
환 및 정규화를 수행하는 것이 중요하다는 것도 알 수 있었습니다. 특히 Term Project 같은 경우,
모델이 훈련 데이터에 과적합될 수 있기에 학습 중에 정확도와 손실 외에도 검증 데이터를 사용
하여 모델의 성능을 평가하고 모니터링하는 것이 Test 정확도와 result.txt 정확도 차이로 나타난다
는 것을 알 수 있었습니다. 수업을 통해서 배웠던 이론들을 Term Project에 직접 적용해보니, 알고
있는 내용들이 더 잘 이해되었던 것 같습니다. 남은 기간 열심히 수업에 임하겠습니다. 감사합니
다.
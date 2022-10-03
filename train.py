import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

from tqdm.auto import tqdm

import albumentations as A  # fast image agumentation library
from albumentations.pytorch.transforms import ToTensorV2  # 이미지 형 변환
import torchvision.models as models
import timm
from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

from model import model as mymodel
from utils.preprocess import train_preprocess
from utils.save_config import *

import warnings
warnings.filterwarnings(action='ignore')

CFG = generate_config(
    "data/config.json",
    IMG_SIZE=224,
    EPOCHS=5,
    LEARNING_RATE=3e-4,
    BATCH_SIZE=4,
    SEED=41
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_loader, val_loader, le = train_preprocess('datasets/train.csv', CFG)


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)  # gpu(cpu)에 적용

    criterion = nn.CrossEntropyLoss().to(device)  # CrossEntropyLoss: 다중분류를 위한 손실함수

    best_score = 0
    best_model = None  # 최고의 모델을 추출하기 위한 파라미터

    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()  # 학습시킴.
        train_loss = []
        for img, label in tqdm(iter(train_loader)):  # train_loader에서 img, text, label 가져옴
            img = img.float().to(device)

            label = label.type(torch.LongTensor)  # label type을 LongTensor로 형변환, 추가하여 에러 해결
            label = label.to(device)

            optimizer.zero_grad()  # 이전 루프에서 .grad에 저장된 값이 다음 루프의 업데이트에도 간섭하는 걸 방지, 0으로 초기화

            model_pred = model(img)  # 예측

            loss = criterion(model_pred, label)  # 예측값과 실제값과의 손실 계산

            # loss = model(images = img, return_loss=True)

            loss.backward()  # .backward() 를 호출하면 역전파가 시작
            optimizer.step()  # optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정

            train_loss.append(loss.item())

        # 모든 train_loss 가져옴
        tr_loss = np.mean(train_loss)

        val_loss, val_score = validation(model, criterion, val_loader, device)  # 검증 시작, 여기서 validation 함수 사용

        print(
            f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step()
            # scheduler의 의미: Learning Rate Scheduler => learning rate를 조절한다.
            # DACON에서는 CosineAnnealingLR 또는 CosineAnnealingWarmRestarts 를 주로 사용한다.

        if best_score < val_score:  # 최고의 val_score을 가진 모델에 대해서만 최종적용을 시킴
            best_score = val_score
            best_model = model

    torch.save(best_model, "model.pt")
    return best_model  # val_score가 가장 높은 모델을 출력


def score_function(real, pred):
    return f1_score(real, pred, average="weighted")


def validation(model, criterion, val_loader, device):
    model.eval()  # nn.Module에서 train time과 eval time에서 수행하는 다른 작업을 수행할 수 있도록 switching 하는 함수

    model_preds = []  # 예측값
    true_labels = []  # 실제값

    val_loss = []

    with torch.no_grad():
        for img, label in tqdm(iter(val_loader)):  # val_loader에서 img, text, label 가져옴
            img = img.float().to(device)
            label = label.type(torch.LongTensor)  # label type을 LongTensor로 형변환, 추가하여 에러 해결
            label = label.to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label)  # 예측값, 실제값으로 손실함수 적용 -> loss 추출

            val_loss.append(loss.item())  # loss 출력, val_loss에 저장

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    test_weighted_f1 = score_function(true_labels, model_preds)  # 실제 라벨값들과 예측한 라벨값들에 대해 f1 점수 계산
    return np.mean(val_loss), test_weighted_f1  # 각각 val_loss, val_score에 적용됨


num_classes = len(le.classes_)

import pprint
pprint.pprint(timm.models.list_models())

# model = mymodel.CustomModel(device, num_classes)
# model = models.mobilenet_v3_large(pretrained=True).to(device)
# model = mymodel.coca(num_classes).to(device)

# model = timm.models.coat_mini(pretrained=True, num_classes=num_classes)      ### 44% 정확도 ###
# model = timm.models.vit_relpos_base_patch16_clsgap_224(pretrained=True, num_classes=num_classes)  ### x<40% 정확도 ###

# model = timm.create_model('swin_large_patch4_window7_224',pretrained=True, num_classes=num_classes).cuda()     ### 1에폭에 39퍼
model = torch.load("model.pt")



summary(model, input_size=(3, 224, 224))

# summary(model, input_size = (3, 448, 448))
# model.eval()
optimizer = torch.optim.SGD(params=model.parameters(), lr=CFG["LEARNING_RATE"], nesterov=True, momentum=0.9)
scheduler = None

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

torch.save(infer_model, "model.pt")
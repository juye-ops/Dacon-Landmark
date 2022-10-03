import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from tqdm.auto import tqdm

import albumentations as A # fast image agumentation library
from albumentations.pytorch.transforms import ToTensorV2 # 이미지 형 변환
import torchvision.models as models

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import json
CFG = json.load(open("data/config.json", "r"))
print(CFG)
test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms, infer=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.infer = infer

    def __getitem__(self, index):
        # Image 읽기
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']  # transforms(=image augmentation) 적용

        # Label
        if self.infer:  # infer == True, test_data로부터 label "결과 추출" 시 사용
            return image
        else:  # infer == False
            label = self.label_list[index]  # dataframe에서 label 가져와 "학습" 시 사용
            return image, label

    def __len__(self):
        return len(self.img_path_list)


test_df = pd.read_csv('datasets/test.csv')
# test_vectors = vectorizer.transform(test_df['overview'])
# test_vectors = test_vectors.todense()

test_dataset = CustomDataset(test_df['img_path'].values, None, test_transform, True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

all_df = pd.read_csv('datasets/train.csv')
train_df, val_df, _, _ = train_test_split(all_df, all_df['cat3'], test_size=0.2, random_state=CFG['SEED'])
le = preprocessing.LabelEncoder()
le.fit(train_df['cat3'].values)

print(len(le.classes_))



def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    model_preds = []

    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)

            model_pred = model(img)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    # img, text에 따른 예측값들을 model_preds 배열에 넣어 리턴
    return model_preds

# model = models.efficientnet_v2_l()
infer_model = torch.load("model.pt")

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['cat3'] = le.inverse_transform(preds)

submit.to_csv('./submit_jgw.csv', index=False)

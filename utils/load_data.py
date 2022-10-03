import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None, infer=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.infer = infer

    def __getitem__(self, index):
        # Image 읽기
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        # image = cv2.resize(image, dsize = (CFG["IMG_SIZE"], CFG["IMG_SIZE"]))

        if self.transforms is not None:
            image = self.transforms(image)  # transforms(=image augmentation) 적용
            image = torch.tensor(image)
        # Label
        if self.infer:  # infer == True, test_data로부터 label "결과 추출" 시 사용
            return image
        else:  # infer == False
            label = self.label_list[index]  # dataframe에서 label 가져와 "학습" 시 사용
            return image, label

    def __len__(self):
        return len(self.img_path_list)


def load_data(dst):
    return pd.read_csv(dst)

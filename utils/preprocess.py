from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

from .load_data import *


def train_preprocess(data_src, config):
    all_df = load_data(data_src)
    train_df, val_df, _, _ = train_test_split(all_df, all_df['cat3'], test_size=0.2, random_state=config['SEED'])

    le = preprocessing.LabelEncoder()
    le.fit(train_df['cat3'].values)

    train_df['cat3'] = le.transform(train_df['cat3'].values)
    val_df['cat3'] = le.transform(val_df['cat3'].values)

    # vectorizer = CountVectorizer(max_features=4096)
    # overview를 vectorize하는 vectorizer 선언, 최대 특성 수는 4096
    #
    # train_vectors = vectorizer.fit_transform(train_df['overview'])
    # train_vectors = train_vectors.todense()
    #
    # val_vectors = vectorizer.transform(val_df['overview'])
    # val_vectors = val_vectors.todense()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((int(config["IMG_SIZE"]*1.5), int(config["IMG_SIZE"]*1.5))),
        transforms.RandomChoice([
            transforms.Resize((config["IMG_SIZE"], config["IMG_SIZE"])),
            transforms.CenterCrop((config["IMG_SIZE"], config["IMG_SIZE"])),
            transforms.RandomCrop((config["IMG_SIZE"], config["IMG_SIZE"])),
        ]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomApply([
            transforms.RandomRotation((-30, 30)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
        ]),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomInvert(0.2),
    ])

    train_dataset = CustomDataset(train_df['img_path'].values, train_df['cat3'].values, transform)
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0)  # 6

    val_dataset = CustomDataset(val_df['img_path'].values, val_df['cat3'].values, transform)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)  # 6

    return train_loader, val_loader, le

if __name__ == "__main__":
    from save_config import *
    CFG = load_config("../data/config.json")
    train_loader, val_loader, le = train_preprocess('../datasets/train.csv', CFG)
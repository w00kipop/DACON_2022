import os
import cv2
import pandas as pd
import torchvision.transforms as transforms

from glob import glob
from torch.utils.data import Dataset, DataLoader


# 수화 Dataset을 생성하는 함수
class SignLanguageDataset(Dataset):


    def __init__(self, img_pth_list, label_list, train_mode=True, transform=True):
        self.img_pth_list = img_pth_list
        self.label_list = label_list
        self.train_mode = train_mode
        self.transform = transform


    # index번째 img를 return한다.
    def __getitem__(self, index):
        img_pth = self.img_pth_list[index]

        # get img data
        img = cv2.imread(img_pth)

        if self.transform is not None: # transform이 None이 아니라면 img에 transform을 적용한다.
            img = self.transform(img)

        if self.train_mode: # 만약 train data라면 img, label을 return
            label = self.label_list[index]
            return img, label
        else:
            return img


    # 데이터의 길이 반환
    def __len__(self):
        return len(self.img_pth_list)



# 데이터를 불러오는 함수
def load_dataset(data_path):
    df = pd.read_csv(data_path) # csv 파일을 받아옴

    df['label'][df['label'] == '10-1'] = 10 # 10-1 -> 10
    df['label'][df['label'] == '10-2'] = 0 # 10-2 -> 0
    df['label'] = df['label'].apply(lambda x: int(x)) # DataType / Object -> Int

    return df


# train_data를 불러오는 함수
def get_train_data(data_path, df):
    img_pth_list = [] # 각 이미지 경로
    label_list = [] # 레이블 리스트

    img_pth_list.extend(glob(os.path.join(data_path, '*.png'))) # img_pth_list에 이미지 파일들을 넣어준다.
    img_pth_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0])) # 이미지들을 오름차순으로 정렬한다.

    # get label
    label_list.extend(df['label'])

    return img_pth_list, label_list


# test_data를 불러오는 함수
def get_test_data(data_path):
    img_pth_list = []

    img_pth_list.extend(glob(os.path.join(data_path, '*.png')))
    img_pth_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))

    return img_pth_list


# transform을 return하는 함수
def get_transform(config, train=True):
    if train:
        train_transform = transforms.Compose([
            transforms.ToPILImage(), # Numpy list -> Image
            transforms.Resize([config.img_size, config.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        return train_transform

    else:
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([config.img_size, config.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        return test_transform


# Img를 split하는 함수
def split_data(config, img_pth_list, label_list):
    train_len = int(len(img_pth_list) * config.train_ratio)

    train_img_pth = img_pth_list[:train_len]
    train_label = label_list[:train_len]

    valid_img_pth = img_pth_list[train_len:]
    valid_label = label_list[train_len:]

    return train_img_pth, train_label, valid_img_pth, valid_label


# dataloader 생성
def get_train_loader(config, train_img_pth, train_label, transform):
    train_dataset = SignLanguageDataset(train_img_pth, train_label, train_mode=True, transform=transform)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=0)

    return train_loader


def get_valid_loader(config, valid_img_pth, valid_label, transform):
    valid_dataset = SignLanguageDataset(valid_img_pth, valid_label, train_mode=True, transform=transform)
    valid_loader = DataLoader(valid_dataset, config.batch_size, shuffle=False, num_workers=0)

    return valid_loader


def get_test_loader(config, test_img_pth, transform):
    test_dataset = SignLanguageDataset(test_img_pth, None, train_mode=False, transform=transform)
    test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False, num_workers=0)

    return test_loader

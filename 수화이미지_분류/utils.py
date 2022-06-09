import os
from tqdm.auto import tqdm
import torch
import random
import pandas as pd
import numpy as np
import matplotlib as plt


# SEED를 고정하는 함수
def fix_seed(config):
    random.seed(config.seed)
    os.environ['PYTHONSEED'] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)


# Train / valid의 이미지 출력
def show_img(data_loader):
    train_features, train_labels = next(iter(data_loader)) # [0]번째 부터 이미지와 label을 가져온다.

    img = train_features[0]
    label = train_labels[0]

    plt.imshow(img[0], cmap='gray') # 흑백 이미지 출력
    plt.show()

    print(f'label: {label}') # label 출력


# 모델의 값을 예측한다.
def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            y_hat = model(img)
            y_hat = y_hat.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(y_hat.tolist())

    return model_pred


# 모델에서 나온 결과를 제출용 csv로 변환한다.
def submission_csv(csv_pth, preds, csv_name):
    submission = pd.read_csv(csv_pth)
    submission['label'] = preds

    submission['label'][submission['label'] == 10] = '10-1'
    submission['label'][submission['label'] == 0] = '10-2'
    submission['label'] = submission['label'].apply(lambda x: str(x)) # int -> object

    submission.to_csv(csv_name, index=False)

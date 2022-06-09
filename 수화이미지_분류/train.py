import argparse

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

from utils import predict
from utils import submission_csv
from trainer import Trainer
from models.resnet import resnet50
from data_loader import load_dataset
from data_loader import get_train_data
from data_loader import get_test_data
from data_loader import split_data
from data_loader import get_transform
from data_loader import get_train_loader
from data_loader import get_valid_loader
from data_loader import get_test_loader

# from utils import fix_seed


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)
    # p.add_argument('--seed', type=int, default=41)

    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--img_size', type=int, default=128)

    p.add_argument('--model', type=str, default='resnet')

    config = p.parse_args()

    return config


def get_model(config):
    if config.model == 'resnet':
        model = resnet50()
    else:
        raise NotImplementedError('You need to specify model name.')

    return model


def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # fix_seed(config.seed)

    df = load_dataset('./data/train.csv')

    img_pth_list, label_list = get_train_data('./data/train', df)
    test_img_pth_list = get_test_data('./data/test')

    train_transform = get_transform(config)
    test_transform = get_transform(config, train=False)
    train_img_pth, train_label, valid_img_pth, valid_label = split_data(config, img_pth_list, label_list)

    train_loader = get_train_loader(config, train_img_pth, train_label, train_transform)
    valid_loader = get_valid_loader(config, valid_img_pth, valid_label, test_transform)
    test_loader = get_test_loader(config, test_img_pth_list, test_transform)

    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, crit, device)
    trainer.train(train_loader, valid_loader, config)
    model.load_state_dict('./model_pth/resnet_best_acc.pth')

    model_pred = predict(model, test_loader, device)
    submission = pd.read_csv('./data/sample_submission.csv')
    submission['label'] = model_pred

    submission['label'][submission['label'] == 10] = '10-1'
    submission['label'][submission['label'] == 0] = '10-2'
    submission['label'] = submission['label'].apply(lambda x: str(x))  # int -> ojbect

    submission.to_csv('resnet50.csv', index=False)


if __name__ == '__main__':
    main_config = define_argparser()
    main(main_config)

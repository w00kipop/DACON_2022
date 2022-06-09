import torch
import numpy as np

from copy import deepcopy
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, optimizer, crit, device):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.device = device

        super().__init__()


    def _train(self, train_loader, config):
        self.model.to(self.device)

        for epoch in range(1, config.epoch + 1):
            self.model.train()
            train_loss = 0.0
            i = 0

            for img, label in tqdm(iter(train_loader)):
                img, label = img.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()

                y_hat = self.model(img)
                loss = self.crit(y_hat, label)

                loss.backward()
                self.optimizer.zero_grad()

                train_loss += loss.item()
                i = i + 1

            print('[%d]Epoch Train loss: %.4f' % (i, train_loss / len(train_loader)))

        return train_loss / len(train_loader)


    def _validate(self, valid_loader, config):
        self.model.to(self.device)
        i = 0

        for epoch in range(1, config.epoch + 1):
            self.model.eval()
            valid_loss = 0.0
            correct = 0
            best_acc = 0.0

            with torch.no_grad():
                for img, label in tqdm(iter(valid_loader)):
                    img,label = img.to(self.device), label.to(self.device)

                    y_hat = self.model(img)
                    loss = self.crit(y_hat, label)

                    valid_loss += loss.item()

                    pred = y_hat.argmax(dim=1, keepdim=True)
                    correct += pred.eq(label.view_as(pred)).sum().item()

            valid_acc = 100 * correct / len(valid_loader.dataset)
            if best_acc < valid_acc:
                i += 1
                best_acc = valid_acc
                torch.save(self.model.state_dict(), './model_pth/resnet_best_acc.pth')
                print(f'best_acc_model이 {i}번 바뀜')

            print('[%d] Valid loss: %.4f, Accuracy: {%d}/{%d} (%.2f)\n' % (config.epoch, valid_loss / len(valid_loader), correct, len(valid_loader.dataset), 100 * correct / len(valid_loader.dataset)))

        return valid_loss / len(valid_loader)


    def train(self, train_loader, valid_loader, config):
        lowest_loss = np.inf
        best_model = None
        i = 0

        for epoch_index in range(config.epoch):
            train_loss = self._train(train_loader, config)
            valid_loss = self._validate(valid_loader, config)

            if valid_loss <= lowest_loss:
                i += 1
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), './model_pth/resnet_lowest_loss.pth')
                print(f'lowest_loss_model이 {i}번 바뀜.')

            print("Epoch(%d/%d): train_loss=%.4e valid_loss=%.4e lowest_loss=%.4e" % (
                epoch_index + 1,
                config.epoch,
                train_loss,
                valid_loss,
                lowest_loss
            ))

        self.model.load_state_dict(best_model)

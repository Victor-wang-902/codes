import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
import numpy as np
from torch.utils.data import IterableDataset

import os
import datetime as dt

from prepare_dataset import load_data_and_random_split


class PredictionDataset(IterableDataset):
    def __init__(self, images):
      self.queue = images

    def read_next_image(self):
        while self.queue.qsize() > 0:
            # you can add transform here
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


class LeNetBasedCNN:
    def __init__(self,
                 train_iter=None,
                 test_iter=None,
                 batch_size=256,
                 num_epochs=20,
                 lr=0.01,
                 train_device=None,
                 test_device=None,
                 NN=None,
                 optim="Adam",
                 wd=0.001,
                 momentum=0.5,
                 dropout=0.2
                 ):
        if NN is None:
            NN = OrderedDict([("Conv1", nn.Conv2d(1, 8, kernel_size=5, padding=2)),
                              ("norm1", nn.BatchNorm2d(num_features=8)),
                              ("ReLU1", nn.ReLU()),
                              ("MaxP1", nn.MaxPool2d(kernel_size=2, stride=2)),
                              ("Cov2", nn.Conv2d(8, 16, kernel_size=5)),
                              ("norm2", nn.BatchNorm2d(num_features=16)),
                              ("ReLU2", nn.ReLU()),
                              ("MaxP2", nn.MaxPool2d(kernel_size=2, stride=2)),
                              ("Flat", nn.Flatten()),
                              ("Linear1", nn.Linear(16 * 5 * 5, 120)),
                              ("Activ1", nn.ReLU()),
                              ("Drop1", nn.Dropout(dropout)),
                              ("Linear2", nn.Linear(120, 84)),
                              ("Activ2", nn.ReLU()),
                              ("Drop2", nn.Dropout(dropout)),
                              ("Output", nn.Linear(84, 47))
                              ])
        self.net = nn.Sequential(NN)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.lr = lr
        self.train_device = train_device
        self.test_device = test_device
        self.batch_size = batch_size
        if optim == "Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=wd)
        elif optim == "SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=wd, momentum=momentum)

    def train(self):
        epochs_train = []
        epochs_test = []
        train_loss = []
        train_acc = []
        test_acc = []
        self.net.apply(init_weights)
        if self.train_device:
            self.net.to(self.train_device)
        else:
            self.train_device = torch.cuda.current_device()
        print('training on', self.train_device)
        loss = nn.CrossEntropyLoss()
        num_batches = len(self.train_iter)
        for epoch in range(self.num_epochs):
            percentage = epoch
            error = 0
            correctness = 0
            num = 0
            self.net.train()
            start_time = dt.datetime.today().timestamp()
            for i, (X, y) in enumerate(self.train_iter):
                self.optimizer.zero_grad()
                X, y = X.to(self.train_device), y.to(self.train_device)
                y_hat = self.net(X)
                L = loss(y_hat, y)
                L.backward()
                self.optimizer.step()
                with torch.no_grad():
                    error += L.item() * X.shape[0]
                    correctness += d2l.accuracy(y_hat, y)
                    num += X.shape[0]
                    current_loss = error / num
                    current_acc = correctness / num
                if i == 0 or i == num_batches // 2:
                    time_diff = dt.datetime.today().timestamp() - start_time
                    print("loss:", current_loss, "accuracy:", current_acc)
                    print("{:.2%} completed".format(percentage / 100 + ((i + 1) / self.batch_size) / num_batches))
                    print("speed:", (i + 1) * self.batch_size / time_diff)
                    train_loss.append(current_loss)
                    train_acc.append(current_acc)
                    if i == 0:
                        epochs_train.append(epoch + 1)
                    else:
                        epochs_train.append(epoch + 1 + 0.5)
            test_accuracy = self.test()
            test_acc.append(test_accuracy)
            print("end of epoch", epoch + 1, "test accuracy:", test_accuracy)
            epochs_test.append(epoch + 1)
        print("training complete")
        self.save()
        return train_loss, train_acc, test_acc

    def validate(self, record_epochs=False):
        test_accuracies = []
        test_epochs = []
        validation_accuracies = []
        validation_losses = []
        validation_epochs = []
        self.net.apply(init_weights)
        if self.train_device:
            self.net.to(self.train_device)
        else:
            self.train_device = torch.cuda.current_device()
        print('validating on', self.train_device)
        loss = nn.CrossEntropyLoss()
        num_batches = len(self.train_iter)
        for epoch in range(self.num_epochs):
            percentage = epoch
            error = 0
            correctness = 0
            num = 0
            self.net.train()
            start_time = dt.datetime.today().timestamp()
            for i, (X, y) in enumerate(self.train_iter):
                self.optimizer.zero_grad()
                X, y = X.to(self.train_device), y.to(self.train_device)
                y_hat = self.net(X)
                L = loss(y_hat, y)
                L.backward()
                self.optimizer.step()
                with torch.no_grad():
                    error += L.item() * X.shape[0]
                    correctness += d2l.accuracy(y_hat, y)
                    num += X.shape[0]
                    current_loss = error / num
                    current_acc = correctness / num
                if i == 0 or i == num_batches // 2:
                    time_diff = dt.datetime.today().timestamp() - start_time
                    print("loss:", current_loss, "accuracy:", current_acc)
                    print("{:.2%} completed".format(percentage / 100 + ((i + 1) / self.batch_size) / num_batches))
                    print("speed:", (i + 1) * self.batch_size / time_diff)
                    if record_epochs:
                        validation_losses.append(current_loss)
                        validation_accuracies.append(current_acc)
                        if i == 0:
                            validation_epochs.append(epoch + 1)
                        else:
                            validation_epochs.append(epoch + 1 + 0.5)
            test_accuracy = self.test()
            if record_epochs:
                test_accuracies.append(test_accuracy)
                test_epochs.append(epoch + 1)
            print("end of epoch", epoch + 1, "test accuracy:", test_accuracy)
        print("validation complete")
        if record_epochs:
            return validation_losses, validation_accuracies, test_accuracies
        else:
            return current_loss, current_acc, test_accuracy

    def test(self):
        correct_prediction = 0
        num_batches = len(self.test_iter)
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()
            if not self.test_device:
                self.test_device = next(iter(self.net.parameters())).device
        else:
            raise Exception("test model unidentified, exiting")
        print("testing on", self.test_device)
        for i, (X, y) in enumerate(self.test_iter):
            mod = 1 if num_batches < 10 else num_batches // 10
            if (i + 1) % mod == 1 or i == num_batches - 1:
                print("{:.2%}".format((i + 1) / num_batches))
            if isinstance(X, list):
                X = [x.to(self.test_device) for x in X]
            else:
                X = X.to(self.test_device)
            Y_hat = self.net(X)
            Y = y.to(self.test_device)
            correct_prediction += d2l.accuracy(Y_hat, Y)
        return correct_prediction / (num_batches * self.batch_size)

    def predict(self, img, path):
        if self.test_device:
            self.net.to(self.test_device)
        else:
            self.test_device = torch.cuda.current_device()
        self.restore(path)
        self.net.eval()
        x = torch.from_numpy(img)
        x.unsqueeze_(0)
        x.unsqueeze_(0)
        x = x.float()
        x = x.to(self.test_device)
        y = self.net(x)
        return y

    def feature_for_predict(self, img):
        if self.test_device:
            self.net.to(self.test_device)
        else:
            self.test_device = torch.cuda.current_device()
        self.net.eval()
        x = torch.from_numpy(img)
        x.unsqueeze_(0)
        x.unsqueeze_(0)
        x = x.float()
        x = x.to(self.test_device)
        x_features = self.net(x)
        return x_features

    def save(self):
        torch.save(self.net, os.path.join(os.getcwd(), "parameters_2.pt"))

    def restore(self, path):
        model = torch.load(path)
        self.net = model

    def feature_extract_mode(self, path):
        self.restore(path)
        self.net = self.net[:-3]

    def feature_extract(self):
        X_train = None
        X_test = None
        Y_train = None
        Y_test = None
        if self.test_device:
            self.net.to(self.test_device)
        else:
            self.test_device = torch.cuda.current_device()
        for i, (X, y) in enumerate(self.train_iter):
            self.net.eval()
            X, y = X.to(self.train_device), y.to(self.train_device)
            x_train = self.net(X)
            x_train = x_train.to("cpu")
            y = y.to("cpu")
            x_train = x_train.detach().numpy()
            y = y.detach().numpy()
            if X_train is None:
                X_train = x_train
                Y_train = y
            else:
                np.append(X_train, x_train)
                np.append(Y_train, y)
        for i, (X, y) in enumerate(self.test_iter):
            self.net.eval()
            X, y = X.to(self.test_device), y.to(self.test_device)
            x_test = self.net(X)
            x_test = x_test.to("cpu")
            y = y.to("cpu")
            x_test = x_test.detach().numpy()
            y = y.detach().numpy()
            if X_test is None:
                X_test = x_test
                Y_test = y
            else:
                np.append(X_test, x_test)
                np.append(Y_test, y)
        return (X_train, Y_train), (X_test, Y_test)


if __name__ == '__main__': # Comment oct.29 I was not using standard nn.module somehow, but there are kinda similar.
    path = os.path.join(os.getcwd(), "vanillaed")
    train_iter, valid_iter, test_iter, labels = load_data_and_random_split(scale=0.1,
                                                                           test_split=0.2,
                                                                           validation_split=0.2,
                                                                           batch_size=256,
                                                                           path=path)
    device = torch.device('cuda')
    net = LeNetBasedCNN(train_iter,
                        test_iter,
                        batch_size=256,
                        train_device=device,
                        test_device=device)
    train_losses, train_accuracies, test_accuracies = net.train()
    plt.subplot(1, 2, 1), plt.plot(np.linspace(1, 20, 40), train_losses, label="train loss")
    plt.subplot(1, 2, 2), plt.plot(np.linspace(1, 20, 40), train_accuracies, label='train accuracy')
    plt.plot(np.linspace(1, 20, 20), test_accuracies, label='test accuracy')
    plt.xticks([]), plt.yticks([])
    plt.legend()
    plt.show()

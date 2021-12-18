from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os

from CNN import LeNetBasedCNN
from prepare_dataset import load_data_and_random_split


def feature_extraction(train_iter, test_iter, pretrained_path):
    device = torch.device('cuda')
    net = LeNetBasedCNN(train_iter, test_iter, train_device=device, test_device=device)
    net.feature_extract_mode(pretrained_path)
    train_data, test_data = net.feature_extract()
    return train_data, test_data


def SVMC_validation(train_data, test_data):
    x = []
    y = []
    #for c in np.linspace(0.1,100,1000):
            #x.append(c)
    for c in np.linspace(0.001, 4, 1000):
            x.append(c)
            clf = svm.SVC(C=c, kernel='rbf', gamma=0.0075, decision_function_shape="ovo")
            clf.fit(train_data[0], train_data[1])
            predicted = clf.predict(test_data[0])
            accuracy = accuracy_score(test_data[1], predicted)
            y.append(accuracy)
            print(c, accuracy)
    plt.plot(x, y)
    plt.show()


def SVMC_train(train_data, test_data, c=2.5, gamma=0.0075, save=False, filename="svm_default.sav"):
    clf = svm.SVC(C=c, kernel='rbf', gamma=gamma, decision_function_shape="ovo")
    clf.fit(train_data[0], train_data[1])
    predicted = clf.predict(test_data[0])
    accuracy = accuracy_score(test_data[1], predicted)
    if save:
        SVMC_save_model(filename, clf)
    return accuracy


def SVMC_predict(features, filename):
    model = SVMC_load_model(filename)
    return model.predict(features)


def SVMC_load_model(filename="SVMC_model.sav"):
    with open(filename, "rb") as handle:
        model = pickle.load(handle)
    return model


def SVMC_save_model(filename, model):
    with open(filename, "wb") as handle:
        pickle.dump(model, handle)

def SVMBC(train_iter, test_iter):
    pass

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "vanillaed")
    train_iter, valid_iter, test_iter, classes = load_data_and_random_split(scale=0.1,
                                                                            validation_split=0.2,
                                                                            test_split=0.2,
                                                                            batch_size=256,
                                                                            path=path)
    pretrained_path = os.path.join(os.getcwd(), "parameters_2.pt")
    train_data, test_data = feature_extraction(train_iter, test_iter, pretrained_path)
    SVMC_train(train_data, test_data, save=True)

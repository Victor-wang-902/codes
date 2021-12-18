import cv2 as cv
import os

import torch
from utils import index_to_str, str_to_label
from SVM import SVMC_predict
from input_image_processing import segmentation_pipeline
from preprocess import vanilla
from CNN import LeNetBasedCNN
import numpy as np
import matplotlib.pyplot as plt


def prediction_method(img, model_path_1, model_path_2=None, mode="CNN"):
    device = torch.device('cuda')
    if mode == "CNN":
        net = LeNetBasedCNN(train_device=device, test_device=device)
        net.restore(model_path_1)
        label = net.predict(img, model_path_1)
        return label
    if mode == "CNN+SVM":
        net = LeNetBasedCNN(train_device=device, test_device=device)
        net.feature_extract_mode(model_path_1)
        x_features = net.feature_for_predict(img)
        x_features = x_features.to("cpu")
        x_features = x_features.detach().numpy()
        label = SVMC_predict(x_features, model_path_2)
        return label


def prediction_pipeline(img_path, model_path_1="parameters_2.pt", model_path_2="SVMC_model.sav", architecture="CNN"):
    lines = segmentation_pipeline(img_path, vanilla)
    print(lines)
    predicted_lines = ""
    for sentence in lines:
        predicted_sentence = ""
        for word in sentence:
            w = ""
            for ch in word:
                plt.imshow(ch)
                plt.show()
                predicted = prediction_method(ch, model_path_1, model_path_2, mode=architecture)
                print(type(predicted))
                if torch.is_tensor(predicted):
                    activation = torch.nn.Softmax(dim=47)
                    output = activation(predicted)
                    output = output.to("cpu")
                    output = output.detach().numpy()
                    print(output)
                w += str_to_label(index_to_str[np.argmax(output)])
            predicted_sentence += w + " "
        predicted_lines += predicted_sentence + "\n"
    return predicted_lines


if __name__ == "__main__":
    img_path = os.path.join(os.getcwd(), "task.png")
    lines = segmentation_pipeline(img_path, vanilla)
#    predicted = prediction_pipeline(img_path, model_path_1="parameters_2.pt", model_path_2="svm_default.sav", architecture="CNN+SVM")
#    print("CNN+SVM:", predicted)
    predicted = prediction_pipeline(img_path, model_path_1="parameters_2.pt", architecture="CNN")
    print("CNN:", predicted)
#    test_img_path = os.path.join(os.getcwd(), "sample_test.png")
#    test_img = cv.imread(test_img_path)
#    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
#    prediction_method(test_img, model_path_1="parameters_2.pt", mode="CNN")


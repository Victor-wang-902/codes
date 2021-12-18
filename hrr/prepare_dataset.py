import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler

from preprocess import preprocess_procedure_1
from utils import str_to_label


def _split(indices, split):
    size = len(indices)
    split = int(np.floor(split * size))
    return indices[split:], indices[:split]


def _balance(subset):
    total = float(len(subset))
    num_classes = len(subset.dataset.classes)
    num_images = len(subset)
    count_per_class = [0] * num_classes
    print("counting observations per class...")
    for i in range(num_images):
        count_per_class[subset[i][1]] += 1.
    weight_per_class = [0.] * num_classes
    print("calculating class-wise weights")
    for i in range(num_classes):
        weight_per_class[i] = total / count_per_class[i]
    weight_per_image = [0.] * num_images
    print("applying weights to observations...")
    for i in range(num_images):
        weight_per_image[i] = weight_per_class[subset[i][1]]
    print("balancing complete")
    return weight_per_image


def load_data_and_random_split(prepare_train=True,
                               prepare_valid=True,
                               prepare_test=True,
                               scale=0.1,
                               test_split=0.2,
                               validation_split=0.2,
                               batch_size=32,
                               path=os.getcwd() + "/by_merge/by_merge"):
    print("initializing...")
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Lambda(lambda img: preprocess_procedure_1(img, 177, 28, 20)),
                                    transforms.ToTensor()]) if path == os.getcwd() + "/by_merge/by_merge" else \
        transforms.Compose([transforms.Grayscale(),
                            transforms.ToTensor()])
    print("reading dataset from directories...")
    dataset = datasets.ImageFolder(path, transform=transform)
    classes = dataset.classes
    total = len(dataset)
    train_num = int(total * (1 - test_split - validation_split) * scale)
    valid_num = int(total * validation_split * scale)
    test_num = int(total * test_split * scale)
    remainder = total - train_num - valid_num - test_num
    if test_split >= 0 and validation_split >= 0 and 1 - test_split - validation_split >= 0:
        print("spliting dataset...")
        train_dataset, validation_dataset, test_dataset, rest = random_split(dataset,
                                                                             [train_num,
                                                                              valid_num,
                                                                              test_num,
                                                                              remainder])
        print("balancing...")
        if prepare_train:
            train_weights = _balance(train_dataset)
        if prepare_valid:
            validation_weights = _balance(validation_dataset)
        if prepare_test:
            test_weights = _balance(test_dataset)
        print("packing dataloader...")
        if prepare_train:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       sampler=WeightedRandomSampler(train_weights,
                                                                                     train_num), drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size)
        if prepare_valid:
            valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                                       batch_size=batch_size,
                                                       sampler=WeightedRandomSampler(validation_weights,
                                                                                     valid_num), drop_last=True)
        else:
            valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                                       batch_size=batch_size, drop_last=True)
        if prepare_test:
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=batch_size,
                                                      sampler=WeightedRandomSampler(test_weights,
                                                                                    test_num), drop_last=True)
        else:
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=batch_size, drop_last=True)
    else:
        raise ValueError("splits should be greater than 0 and sum of splits smaller than 1")
    return train_loader, valid_loader, test_loader, classes


def get_image_sample(dataloader, classes):
    images, indices = next(iter(dataloader))
    print(str_to_label(classes, indices[0]))
    plt.imshow(images[0][0], cmap='gray', vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "vanillaed")
    train_iter, valid_iter, test_iter, classes = load_data_and_random_split(scale=0.01,
                                                                            validation_split=0.2,
                                                                            test_split=0.2,
                                                                            batch_size=256,
                                                                            path=path)
    print("checking the balacing of the dataset...")
    print("total number of observations:", len(train_iter) * 256)
    class_count = [0] * len(classes)
    for X, y in train_iter:
        for i in y:
            class_count[i] += 1
    for i in range(len(class_count)):
        print("num of {}: {}".format(str_to_label(classes[i]), class_count[i]))

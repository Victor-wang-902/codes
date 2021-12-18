import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle
from collections import OrderedDict
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task

from prepare_dataset import load_data_and_random_split
from CNN import LeNetBasedCNN

BATCH_SIZE = 0
LEARNING_RATE = 1
STRUCTURE = 2
EPOCH_SIZE = 3
OPTIMIZER = 4
DATA_SIZE = 5


def validate_epoch(path, parameters):
    device = torch.device("cuda")
    validation_losses = [0] * len(parameters)
    validation_accuracies = [0] * len(parameters)
    test_accuracies = [0] * len(parameters)
    ind = 0
    train_iter, validation_iter, test_iter, labels = load_data_and_random_split(scale=0.01,
                                                                                test_split=0.2,
                                                                                validation_split=0.2,
                                                                                batch_size=256,
                                                                                path=path)
    net = LeNetBasedCNN(validation_iter,
                        test_iter,
                        batch_size=256,
                        num_epochs=parameters[-1],
                        train_device=device,
                        test_device=device)
    epochs = net.num_epochs
    model_validation_losses, model_validation_accuracies, model_test_accuracies = net.validate(record_epochs=True)
    for parameter in parameters:
        validation_losses[ind] = model_validation_losses[parameter * 2 - 1]
        validation_accuracies[ind] = model_validation_accuracies[parameter * 2 - 1]
        test_accuracies[ind] = model_test_accuracies[parameter - 1]
        ind += 1
    print(validation_losses, validation_accuracies, test_accuracies)
    return validation_losses, validation_accuracies, test_accuracies


def validate(path, parameter_type=BATCH_SIZE, parameters=range(5, 9)):
    device = torch.device("cuda")
    validation_losses = [0] * len(parameters)
    validation_accuracies = [0] * len(parameters)
    test_accuracies = [0] * len(parameters)
    ind = 0
    train_iter, validation_iter, test_iter, labels = load_data_and_random_split(scale=0.01,
                                                                                test_split=0.2,
                                                                                validation_split=0.2,
                                                                                batch_size=256,
                                                                                path=path,
                                                                                )
    if parameter_type == BATCH_SIZE:
        for parameter in parameters:
            train_iter, validation_iter, test_iter, labels = load_data_and_random_split(scale=0.25,
                                                                                        test_split=0.2,
                                                                                        validation_split=0.2,
                                                                                        batch_size=parameter,
                                                                                        path=path)
            net = LeNetBasedCNN(validation_iter,
                                test_iter,
                                batch_size=parameter,
                                train_device=device,
                                test_device=device)
            model_validation_losses, model_validation_accuracies, model_test_accuracies = net.validate(
                record_epochs=True)
            validation_losses[ind] = model_validation_losses
            validation_accuracies[ind] = model_validation_accuracies
            test_accuracies[ind] = model_test_accuracies
            ind += 1
    elif parameter_type == LEARNING_RATE:
        for parameter in parameters:
            net = LeNetBasedCNN(validation_iter,
                                test_iter,
                                batch_size=256,
                                lr=parameter,
                                train_device=device,
                                test_device=device)
            validation_loss, validation_acc, test_acc = net.validate()
            validation_losses[ind] = validation_loss
            validation_accuracies[ind] = validation_acc
            test_accuracies[ind] = test_acc
            ind += 1
    elif parameter_type == STRUCTURE:
        title = 'tuning structure'
        xlabel = 'structure type'
        for parameter in parameters:
            net = LeNetBasedCNN(validation_iter,
                                test_iter,
                                batch_size=256,
                                train_device=device,
                                test_device=device,
                                NN=parameter)
            validation_loss, validation_acc, test_acc = net.validate(record_epochs=False)
            validation_losses[ind] = validation_loss
            validation_accuracies[ind] = validation_acc
            test_accuracies[ind] = test_acc
            ind += 1
    elif parameter_type == OPTIMIZER:
        title = "tuning optimizer"
        xlabel = "optimizer"
        for parameter in parameters:
            if parameter == "Adam":
                net = LeNetBasedCNN(validation_iter,
                                    test_iter,
                                    batch_size=256,
                                    train_device=device,
                                    test_device=device,
                                    optim=parameter)
                validation_loss, validation_acc, test_acc = net.validate(record_epochs=True)
                validation_losses[ind] = validation_loss
                validation_accuracies[ind] = validation_acc
                test_accuracies[ind] = test_acc
                ind += 1
            elif parameter == "SGD1":
                net = LeNetBasedCNN(validation_iter,
                                    test_iter,
                                    batch_size=256,
                                    train_device=device,
                                    test_device=device,
                                    optim=parameter)
                validation_loss, validation_acc, test_acc = net.validate(record_epochs=True)
                validation_losses[ind] = validation_loss
                validation_accuracies[ind] = validation_acc
                test_accuracies[ind] = test_acc
                ind += 1
            elif parameter == "SGD2":
                net = LeNetBasedCNN(validation_iter,
                                    test_iter,
                                    batch_size=256,
                                    train_device=device,
                                    test_device=device,
                                    optim=parameter,
                                    momentum=0.9)
                validation_loss, validation_acc, test_acc = net.validate(record_epochs=True)
                validation_losses[ind] = validation_loss
                validation_accuracies[ind] = validation_acc
                test_accuracies[ind] = test_acc
                ind += 1
    else:
        raise Exception("wrong hyperparameter")
    return validation_losses, validation_accuracies, test_accuracies


def save_result(data_path, parameters, parameter_type=BATCH_SIZE):
    if parameter_type == LEARNING_RATE:
        filename = "learning_rate_tuning.pickle"
    elif parameter_type == BATCH_SIZE:
        filename = "batch_size_tuning2.pickle"
    elif parameter_type == STRUCTURE:
        filename = "structure_tuning2.pickle"
    elif parameter_type == EPOCH_SIZE:
        filename = "epoch_size_tuning2.pickle"
    elif parameter_type == OPTIMIZER:
        filename = "optimizer_tuning.pickle"
    else:
        raise Exception("wrong parameter type")
    if parameter_type == EPOCH_SIZE:
        validation_losses, validation_accuracies, test_accuracies = validate_epoch(data_path, parameters)
    else:
        validation_losses, validation_accuracies, test_accuracies = validate(data_path,
                                                                             parameter_type=parameter_type,
                                                                             parameters=parameters)
    package = [validation_losses, validation_accuracies, test_accuracies, parameters]
    with open(filename, "wb") as handle:
        print("writing file...")
        pickle.dump(package, handle)


def plot(filename):
    with open(filename, "rb") as handle:
        package = pickle.load(handle)
    print(package)
    if isinstance(package[0][0], list):
        epochs = list(range(1, len(package[0][0]) + 1))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title("validation loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ind = 0
        for item in package[0]:
            ax1.plot(epochs, item, label=package[5][ind])
            ind += 1
        ax2.set_title("validation accuracy")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy")
        ind = 0
        for item in package[1]:
            ax2.plot(epochs, item, label=package[5][ind])
            ind += 1
        ax3.set_title("test accuracy")
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("accuracy")
        ind = 0
        for item in package[2]:
            ax3.plot(epochs, item, label=package[5][ind])
            ind += 1
        fig.suptitle(package[3])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(package[3])
        ax1.set_xscale("log")
        ax1.set_xlabel(package[4])
        ax1.set_ylabel("loss")
        ax1.set_title("validation loss")
        ax1.plot(package[5], package[0], label="validation loss")
        ax2.set_xlabel(package[4])
        ax2.set_ylabel("accuracy")
        ax2.set_title("validation & test accuracy")
        ax2.set_xscale("log")
        ax2.plot(package[5], package[1], label="validation accuracy")
        ax2.plot(package[5], package[2], label="test accuracy")
    plt.legend()
    plt.show()


def tune_lr(data_path, parameters=None):
    if parameters is None:
        parameters = [0.1, 0.01, 0.001, 0.0001]
    print("commencing learning rate tuning...")
    save_result(data_path, parameters, parameter_type=LEARNING_RATE)


def tune_bs(data_path, parameters=None):
    if parameters is None:
        parameters = [32, 64, 128, 256, 512, 1024]
    print("commencing batch size tuning...")
    save_result(data_path, parameters, parameter_type=BATCH_SIZE)


def tune_epoch(data_path, parameters=None):
    if parameters is None:
        parameters = [1] #,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print("commencing epoch size tuning...")
    save_result(data_path, parameters, parameter_type=EPOCH_SIZE)


def tune_structure(data_path, parameters=None):
    if parameters is None:
        parameters = [OrderedDict([("Conv1", nn.Conv2d(1, 6, kernel_size=5, padding=2)),
                                   ("norm1", nn.BatchNorm2d(num_features=6)),
                                   ("ReLU1", nn.ReLU()),
                                   ("MaxP1", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Cov2", nn.Conv2d(6, 16, kernel_size=5)),
                                   ("norm2", nn.BatchNorm2d(num_features=16)),
                                   ("ReLU2", nn.ReLU()),
                                   ("MaxP2", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Flat", nn.Flatten()),
                                   ("Linear1", nn.Linear(16 * 5 * 5, 120)),
                                   ("Activ1", nn.Sigmoid()),
                                   ("Linear2", nn.Linear(120, 84)),
                                   ("Activ2", nn.Sigmoid()),
                                   ("Linear3", nn.Linear(84, 47))
                                   ]),
                      '''OrderedDict([("Conv1", nn.Conv2d(1, 6, kernel_size=5, padding=2)),
                                   ("norm1", nn.BatchNorm2d(num_features=6)),
                                   ("ReLU1", nn.ReLU()),
                                   ("MaxP1", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Cov2", nn.Conv2d(6, 16, kernel_size=5)),
                                   ("norm2", nn.BatchNorm2d(num_features=16)),
                                   ("ReLU2", nn.ReLU()),
                                   ("MaxP2", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Flat", nn.Flatten()),
                                   ("Linear1", nn.Linear(16 * 5 * 5, 120)),
                                   ("Activ1", nn.Sigmoid()),
                                   ("Drop1", nn.Dropout(0.2)),
                                   ("Linear2", nn.Linear(120, 84)),
                                   ("Activ2", nn.Sigmoid()),
                                   ("Drop2", nn.Dropout(0.2)),
                                   ("Linear3", nn.Linear(84, 47)),
                                   ("Activ3", nn.Sigmoid()),
                                   ("Drop3", nn.Dropout(0.2)),
                                   ("Linear4", nn.Linear(47, 37))
                                   ]),
                      OrderedDict([("Conv1", nn.Conv2d(1, 6, kernel_size=5, padding=2)),
                                   ("norm1", nn.BatchNorm2d(num_features=6)),
                                   ("ReLU1", nn.ReLU()),
                                   ("MaxP1", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Cov2", nn.Conv2d(6, 16, kernel_size=5)),
                                   ("norm2", nn.BatchNorm2d(num_features=16)),
                                   ("ReLU2", nn.ReLU()),
                                   ("MaxP2", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Flat", nn.Flatten()),
                                   ("Linear1", nn.Linear(16 * 5 * 5, 120)),
                                   ("Activ1", nn.Sigmoid()),
                                   ("Drop1", nn.Dropout(0.2)),
                                   ("Linear2", nn.Linear(120, 84)),
                                   ]),
                      OrderedDict([("Conv1", nn.Conv2d(1, 6, kernel_size=3, padding=1)),
                                   ("norm1", nn.BatchNorm2d(num_features=6)),
                                   ("ReLU1", nn.ReLU()),
                                   ("MaxP1", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Cov2", nn.Conv2d(6, 16, kernel_size=3)),
                                   ("norm2", nn.BatchNorm2d(num_features=16)),
                                   ("ReLU2", nn.ReLU()),
                                   ("MaxP2", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Cov3", nn.Conv2d(16, 32, kernel_size=3)),
                                   ("norm3", nn.BatchNorm2d(num_features=32)),
                                   ("ReLU3", nn.ReLU()),
                                   ("MaxP3", nn.MaxPool2d(kernel_size=2, stride=2)),
                                   ("Flat", nn.Flatten()),
                                   ("Linear1", nn.Linear(32 * 2 * 2, 240)),
                                   ("Activ1", nn.Sigmoid()),
                                   ("Linear2", nn.Linear(240, 120)),
                                   ("Activ2", nn.Sigmoid()),
                                   ("Linear3", nn.Linear(120, 84)),
                                   ("Activ3", nn.Sigmoid()),
                                   ("Linear4", nn.Linear(84, 47))
                                   ])''']
    print("commencing optimizer tuning...")
    save_result(data_path, parameters, parameter_type=STRUCTURE)


def tune_optim(data_path, parameters=None):
    if parameters is None:
        parameters = ["Adam", "SGD1", "SGD2"]
    print("commencing optimizer tuning...")
    save_result(data_path, parameters, parameter_type=OPTIMIZER)


def tune_data_size(data_path, parameters=None):
    if parameters is None:
        parameters = [0.01, 0.05, 0.1, 0.25, 0.5]
    print("commencing optimizer tuning...")
    save_result(data_path, parameters, parameter_type=DATA_SIZE)


if __name__ == "__main__":
    # model_snapshots_path = os.path.join(os.getcwd(), 'clearml')
    # if not os.path.exists(model_snapshots_path):
    #   os.makedirs(model_snapshots_path)
    # task = Task.init(project_name="hrr",
    #                task_name="optimization",
    #               task_type=Task.TaskTypes.optimizer)
    data_path = os.path.join(os.getcwd(), "vanillaed")
    tune_structure(data_path)
    tune_bs(data_path)
    tune_optim(data_path)
    tune_lr(data_path)
    tune_bs(data_path)
    tune_epoch(data_path)

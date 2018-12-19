import torch.nn as nn
import torch
import os
import torchvision
import torch.utils.data
import torch.optim as optim
import time

PATH = os.path.dirname(os.path.abspath(__file__))
PATH_STATE = os.path.join(PATH, "state")
default_name = "deep.mod"
PATH_LOG = os.path.join(PATH, "log")
default_learning_rate = 0.0001
default_momentum = 0.9

OPT = {
    "adam": optim.Adam,
    "sgd" : optim.SGD
}


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                model_name=default_name, path_state=PATH_STATE,  path_log=PATH_LOG,
                learning_rate=default_learning_rate, momentum=default_momentum, optimize_func = optim.SGD,
                epoch_num=2, device=None, default_check=100,):
    """
    :param model: nn.Module
    :param train_loader: train data set in DataLoader
    :param model_name: Name of model
    :param path_state: path folder for storing trained model
    :param path_log: path to write the log file. Log file will be written by model name

    :param learning_rate is the the learning rate for optimize
    :param momentum is the momentum used for stochastic gradient descent -> others -> set to None
    :param optimize_func is the function used for optimized

    :param epoch_num: number of epoch
    :param device: "cpu" or "cuda:0"
    :param default_check: check point every 100
    :return: trained model
    """
    file_model = model_name + ".bin"
    file_log = model_name + ".log"
    file_summary = model_name + "_sum.log"
    save_file = os.path.join(path_state, file_model)
    save_log = os.path.join(path_log, file_log)
    save_log2000 = os.path.join(path_log, file_summary)
    f = open(save_log, "w")
    f1 = open(save_log2000, "w")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if momentum is not None:
        optimizer = optimize_func(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optimize_func(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        running_loss = 0.0
        running_check = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            # feed-forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            end_time = time.time() - start_time
            # print("[%d, %d]: loss: %f"%(epoch, i, float(running_loss/(i+1))))


            # statistic
            loss_item = loss.item()
            running_loss += loss_item
            running_check += loss_item

            print("[%d, %d]: loss: %f - time %f" % (epoch, i, float(running_loss / (i + 1)), end_time))
            f.write("[%d, %d]: loss: %f\n" % (epoch, i, float(running_loss / (i + 1))))
            if i % default_check == default_check-1:
                print("[%d, %5d]: loss: %.3f"%(epoch, i+1, running_check/default_check))
                f1.write("[%d, %5d]: loss: %.3f\n"%(epoch, i+1, running_check/default_check))
                running_check=0.0
    print("Finished training")
    torch.save(model.state_dict(), save_file)
    f.close()
    f1.close()
    return model


def load_data(model: nn.Module, model_name="model", path=PATH_STATE, device=None):
    if not model_exist(model_name, path):
        return None
    model_name += ".bin"
    save_file = os.path.join(path, model_name)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_file, map_location=device))
    model.to(device)
    return model


def model_exist(model_name, path=PATH_STATE):
    save_file = os.path.join(path, model_name+".bin")
    return os.path.isfile(save_file)


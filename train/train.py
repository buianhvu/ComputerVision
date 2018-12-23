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
DEFAULT_STATE_DICT_MODEL = "model-state-dict"
DEFAULT_STATE_DICT_OPT = "optimizer-state-dict"
EPOCH_STR = "epoch"
LOSS_STR = "loss"

LOG_EXT = ".log"
STATE_EXT = ".bin"
LOSS_ADD = "_loss"
LOSS_SUMMARY = "_loss_summary"
ACC_ADD = "_accuracy"


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


def load_eval_model(model: nn.Module, model_name="model", path=PATH_STATE, device=None):
    if not model_exist(model_name, path):
        return None
    model_name += ".bin"
    save_file = os.path.join(path, model_name)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_file, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_training_model(model: nn.Module, optimizer: optim.Optimizer, model_name="model", path_state=PATH_STATE, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_file = model_name + STATE_EXT
    state_file = os.path.join(path_state, state_file)
    checkpoint = torch.load(state_file)
    model.load_state_dict(checkpoint[DEFAULT_STATE_DICT_MODEL])
    optimizer.load_state_dict(checkpoint[DEFAULT_STATE_DICT_OPT])
    epoch = checkpoint['epoch']
    model.to(device)
    model.train()
    return model, optimizer, epoch
    pass


def model_exist(model_name, path=PATH_STATE):
    save_file = os.path.join(path, model_name+".bin")
    return os.path.isfile(save_file)


def calculate_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_correct = 0.
    total = 0.
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            n = len(images)
            output = model(images)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            for i in range(n):
                total_correct += c[i].item()
                total += 1
    return float(total_correct/total)
    pass


def train_model_ac(model: nn.Module, train_loader: torch.utils.data.DataLoader,  # tested
                model_name=default_name, path_state=PATH_STATE,  path_log=PATH_LOG,
                learning_rate=default_learning_rate, momentum=default_momentum, optimize_func = optim.SGD,
                epoch_num=2, device=None, default_check=100):
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
    file_log = model_name + "_loss.log"
    file_summary = model_name + "_loss_summary.log"
    file_accuracy = model_name + "_accuracy.log"
    save_file = os.path.join(path_state, file_model)
    save_log = os.path.join(path_log, file_log)
    save_log2000 = os.path.join(path_log, file_summary)
    save_acc = os.path.join(path_log, file_accuracy)
    f = open(save_log, "w")
    f1 = open(save_log2000, "w")
    f2 = open(save_acc, "w")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # set momentum
    if momentum is not None:
        optimizer = optimize_func(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optimize_func(model.parameters(), lr=learning_rate)

    # run all in epoch_num
    for epoch in range(epoch_num):
        running_loss = 0.0
        running_check = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            b_size = len(inputs)
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

            print("[%d, %d]: loss: %f - average loss: %f - time %f" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1)), end_time))
            f.write("[%d, %d]: loss: %f - average loss : %f\n" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1))))
            if i % default_check == default_check-1:
                print("[%d, %5d]: loss: %.3f"%(epoch, i+1, running_check/default_check))
                f1.write("[%d, %5d]: loss: %.3f\n"%(epoch, i+1, running_check/default_check))
                running_check=0.0
        acc = calculate_accuracy(model, train_loader, device=device)
        f2.write("epoch %2d: %f\n"%(epoch, acc))
        print("epoch %2d: %f"%(epoch, acc))

    print("Finished training")
    torch.save(model.state_dict(), save_file)
    f.close()
    f1.close()
    f2.close()
    return model


def clear_path(model_name: str, path_state: str=PATH_STATE, path_log: str=PATH_LOG):
    """
    Clear all files before training
    :param model_name: name of model to clear
    :param path_state: path to place to save states
    :param path_log: path to palce to save log
    :return:
    """
    # create file name
    file_model = model_name + STATE_EXT  # adding extension .bin
    file_log = model_name + LOSS_ADD + LOG_EXT  # adding _loss.log
    file_summary = model_name + LOSS_SUMMARY + LOG_EXT  # adding _loss_summary.log
    file_accuracy = model_name + ACC_ADD + LOG_EXT  # adding _accuracy.log

    # create file path
    save_states = os.path.join(path_state, file_model)
    save_loss = os.path.join(path_log, file_log)
    save_loss_summary = os.path.join(path_log, file_summary)
    save_acc = os.path.join(path_log, file_accuracy)

    # delete bin file:
    os.remove(save_states)
    # open to append to file
    f = open(save_loss, "w")
    f1 = open(save_loss_summary, "w")
    f2 = open(save_acc, "w")
    f.close()
    f1.close()
    f2.close()


def train_model_ac_load_save(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                model_name=default_name, path_state=PATH_STATE,  path_log=PATH_LOG,
                learning_rate=default_learning_rate, momentum=default_momentum, optimize_func = optim.SGD,
                epoch_num=2, device=None, default_check=100):
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
    # create file name
    file_model = model_name + STATE_EXT  # adding extension .bin
    file_log = model_name + LOSS_ADD + LOG_EXT  # adding _loss.log
    file_summary = model_name + LOSS_SUMMARY + LOG_EXT  # adding _loss_summary.log
    file_accuracy = model_name + ACC_ADD + LOG_EXT # adding _accuracy.log

    # create file path
    save_states = os.path.join(path_state, file_model)
    save_loss = os.path.join(path_log, file_log)
    save_loss_summary = os.path.join(path_log, file_summary)
    save_acc = os.path.join(path_log, file_accuracy)

    # open to append to file
    f = open(save_loss, "a")
    f1 = open(save_loss_summary, "a")
    f2 = open(save_acc, "a")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # setup optimizer
    if momentum is not None:
        optimizer = optimize_func(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optimize_func(model.parameters(), lr=learning_rate)

    # run on epoch
    epoch = 0
    while epoch < epoch_num:
        if model_exist(model_name, path_state):
            model, optimizer, epoch = load_training_model(model, optimizer,
                                                          model_name=model_name, device=device,
                                                          path_state=path_state)
            epoch += 1
            if epoch >= epoch_num:
                break
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

            # statistic
            loss_item = loss.item()
            running_loss += loss_item
            running_check += loss_item
            print("[%d, %d]: loss: %f - average loss: %f - time %f" % (epoch, i, float(loss_item), float(running_loss / (i + 1)), end_time))
            f.write("[%d, %d]: loss: %f - average loss : %f\n" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1))))
            if i % default_check == default_check-1:
                print("[%d, %5d]: loss: %.3f"%(epoch, i+1, running_check/default_check))
                f1.write("[%d, %5d]: loss: %.3f\n"%(epoch, i+1, running_check/default_check))
                running_check=0.0

        # save to file
        torch.save({
            EPOCH_STR: epoch,
            DEFAULT_STATE_DICT_MODEL: model.state_dict(),
            DEFAULT_STATE_DICT_OPT: optimizer.state_dict(),
        }, save_states)
        acc = calculate_accuracy(model, train_loader, device=device)
        f2.write("epoch %2d: %f\n"%(epoch, acc))
        print("epoch %2d: %f"%(epoch, acc))

    #  Finally saved
    print("Finished training")
    torch.save(model.state_dict(), save_states)
    f.close()
    f1.close()
    f2.close()
    return model






# import torch.nn as nn
# import torch
# import os
# import numpy as np
# import torch.utils.data
# import torch.optim as optim
# import time
#
# PATH = os.path.dirname(os.path.abspath(__file__))
# PATH_STATE = os.path.join(PATH, "state")
# default_name = "deep.mod"
# PATH_LOG = os.path.join(PATH, "log")
# default_learning_rate = 0.0001
# default_momentum = 0.9
#
# OPT = {
#     "adam": optim.Adam,
#     "sgd" : optim.SGD
# }
#
#
# def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
#                 model_name=default_name, path_state=PATH_STATE,  path_log=PATH_LOG,
#                 learning_rate=default_learning_rate, momentum=default_momentum, optimize_func = optim.SGD,
#                 epoch_num=2, device=None, default_check=100):
#     """
#     :param model: nn.Module
#     :param train_loader: train data set in DataLoader
#     :param model_name: Name of model
#     :param path_state: path folder for storing trained model
#     :param path_log: path to write the log file. Log file will be written by model name
#
#     :param learning_rate is the the learning rate for optimize
#     :param momentum is the momentum used for stochastic gradient descent -> others -> set to None
#     :param optimize_func is the function used for optimized
#
#     :param epoch_num: number of epoch
#     :param device: "cpu" or "cuda:0"
#     :param default_check: check point every 100
#     :return: trained model
#     """
#     file_model = model_name + ".bin"
#     file_log = model_name + ".log"
#     file_summary = model_name + "_sum.log"
#     save_file = os.path.join(path_state, file_model)
#     save_log = os.path.join(path_log, file_log)
#     save_log2000 = os.path.join(path_log, file_summary)
#     f = open(save_log, "w")
#     f1 = open(save_log2000, "w")
#
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     if momentum is not None:
#         optimizer = optimize_func(model.parameters(), lr=learning_rate, momentum=momentum)
#     else:
#         optimizer = optimize_func(model.parameters(), lr=learning_rate)
#
#     for epoch in range(epoch_num):
#         running_loss = 0.0
#         running_check = 0.0
#         for i, data in enumerate(train_loader):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             start_time = time.time()
#             # feed-forward + backward + optimize
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             end_time = time.time() - start_time
#             # print("[%d, %d]: loss: %f"%(epoch, i, float(running_loss/(i+1))))
#
#
#             # statistic
#             loss_item = loss.item()
#             running_loss += loss_item
#             running_check += loss_item
#
#             print("[%d, %d]: loss: %f - average loss: %f - time %f" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1)), end_time))
#             f.write("[%d, %d]: loss: %f - average loss : %f\n" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1))))
#             if i % default_check == default_check-1:
#                 print("[%d, %5d]: loss: %.3f"%(epoch, i+1, running_check/default_check))
#                 f1.write("[%d, %5d]: loss: %.3f\n"%(epoch, i+1, running_check/default_check))
#                 running_check=0.0
#     print("Finished training")
#     torch.save(model.state_dict(), save_file)
#     f.close()
#     f1.close()
#     return model
#
#
# def load_evaluation_model(model: nn.Module, model_name="model", path=PATH_STATE, device=None):
#     if not model_exist(model_name, path):
#         return None
#     model_name += ".bin"
#     save_file = os.path.join(path, model_name)
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.load_state_dict(torch.load(save_file))
#     model.to(device)
#     model.eval()
#     return model
#
#
# def model_exist(model_name, path=PATH_STATE):
#     save_file = os.path.join(path, model_name+".bin")
#     return os.path.isfile(save_file)
#
#
# def train_model_ac(model: nn.Module, train_loader: torch.utils.data.DataLoader,
#                 model_name=default_name, path_state=PATH_STATE,  path_log=PATH_LOG,
#                 learning_rate=default_learning_rate, momentum=default_momentum, optimize_func = optim.SGD,
#                 epoch_num=2, device=None, default_check=100):
#     """
#     :param model: nn.Module
#     :param train_loader: train data set in DataLoader
#     :param model_name: Name of model
#     :param path_state: path folder for storing trained model
#     :param path_log: path to write the log file. Log file will be written by model name
#
#     :param learning_rate is the the learning rate for optimize
#     :param momentum is the momentum used for stochastic gradient descent -> others -> set to None
#     :param optimize_func is the function used for optimized
#
#     :param epoch_num: number of epoch
#     :param device: "cpu" or "cuda:0"
#     :param default_check: check point every 100
#     :return: trained model
#     """
#     file_model = model_name + ".bin"
#     file_log = model_name + ".log"
#     file_summary = model_name + "_check.log"
#     save_file = os.path.join(path_state, file_model)
#     save_log = os.path.join(path_log, file_log)
#     save_log2000 = os.path.join(path_log, file_summary)
#     f = open(save_log, "w")
#     f1 = open(save_log2000, "w")
#
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     if momentum is not None:
#         optimizer = optimize_func(model.parameters(), lr=learning_rate, momentum=momentum)
#     else:
#         optimizer = optimize_func(model.parameters(), lr=learning_rate)
#
#     for epoch in range(epoch_num):
#         running_loss = 0.0
#         running_check = 0.0
#         for i, data in enumerate(train_loader):
#             inputs, labels = data
#             b_size = len(inputs)
#             inputs, labels = inputs.to(device), labels.to(device)
#             start_time = time.time()
#             # feed-forward + backward + optimize
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             end_time = time.time() - start_time
#             # print("[%d, %d]: loss: %f"%(epoch, i, float(running_loss/(i+1))))
#
#
#             # statistic
#             loss_item = loss.item()
#             running_loss += loss_item
#             running_check += loss_item
#
#             print("[%d, %d]: loss: %f - average loss: %f - time %f" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1)), end_time))
#             f.write("[%d, %d]: loss: %f - average loss : %f\n" % (epoch, i, float(loss_item) ,float(running_loss / (i + 1))))
#             if i % default_check == default_check-1:
#                 print("[%d, %5d]: loss: %.3f"%(epoch, i+1, running_check/default_check))
#                 f1.write("[%d, %5d]: loss: %.3f\n"%(epoch, i+1, running_check/default_check))
#                 running_check=0.0
#     print("Finished training")
#     torch.save(model.state_dict(), save_file)
#     f.close()
#     f1.close()
#     return model

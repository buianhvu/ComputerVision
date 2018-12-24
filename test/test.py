import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from test import test_log


def test_data(model: nn.Module, test_loader: DataLoader, classes, device=None, write_log=True, file_log="test.log"):
    num_classes = len(classes)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_correct = 0.
    total = 0.
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            n = len(images)
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            for i in range(n):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                total_correct += c[i].item()
                total += 1

    for i in range(num_classes):
        print('Accuracy of %5s: %2d %%' %(classes[i], 100 * class_correct[i]/class_total[i]))
    accuracy = total_correct/total
    print('Accuracy: %2d %%' % (accuracy*100))
    if write_log:
        test_log.write_log(classes, class_correct, class_total, accuracy, path=file_log)


def test_unknown(model: nn.Module, test_loader: DataLoader, device=None, write_log=True):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_correct = {}
    class_total = {}
    total_correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            n = len(images)
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            for i in range(n):
                label = int(labels[i])
                if label not in class_correct:
                    class_correct[label] = 0.
                if label not in class_total:
                    class_total[label] = 0.
                class_correct[label] += c[i].item()
                class_total[label] += 1
                total_correct += c[i].item()
                total += 1
    num_classes = len(class_total)
    for key in class_total:
        print('Accuracy of %d: %2d %%' % (key, 100 * class_correct[key] / class_total[key]))
    accuracy = total_correct/total
    print('Accuracy: %2d%%' % (accuracy*100))
    if write_log:
        test_log.write_log_unknown(class_correct, class_total, accuracy)

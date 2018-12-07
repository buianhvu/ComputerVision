import torch
from torch.utils import data
import glob, os
from model.vgg import *
import torch.nn as nn
from torchvision import transforms
from commons.cv_input import *

# Hyper-parameters
input_size = 512*512
hidden_size = 500
num_classes = 11
num_epochs = 5
batch_size = 4
number_of_workers = 4
shuffled = True
learning_rate = 0.001
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


max_epochs = 100

training_set, training_generator = get_loader() # default is used for training
validation_set, validation_generator = get_loader(inside="validation", target_type='val')
test_set, test_generator = get_loader(inside="evaluation", target_type='test')

model = set_vgg16(num_classes=10)
# torch.nn.DataParallel(model.features)
model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(training_generator)

for images, labels in test_generator:
    print(labels)
    print(str(len(labels)))





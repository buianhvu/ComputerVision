from data_loading import Dataset, Rescale, ToTensor, RandomCrop
import torch
from torch.utils import data
import glob,os
import vgg
import torch.nn as nn
from torchvision import transforms, utils

import matplotlib.pyplot as plt
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
# cudnn.benchmark = True

# Parameters

max_epochs = 100
def get_ids_labels(dir, ids_list, lab_dict):
    os.chdir(dir)
    for file in glob.glob("*.jpg"):
        name = file[:-4]
        lab_dict[name] = int(file[0])
        ids_list.append(name)
    #back to root folder
    os.chdir("..")
    os.chdir("..")

train_ids = []
val_ids = []
test_ids = []
train_labels = {}
val_labels = {}
test_labels = {}

#read_filenames
# print(os.getcwd())
get_ids_labels("Food-11/training", train_ids, train_labels)
get_ids_labels("Food-11/validation", val_ids, val_labels)
get_ids_labels("Food-11/evaluation", test_ids, test_labels)

# Generators

training_set = Dataset(train_ids, train_labels, 'train', transform=transforms.Compose([Rescale((512,512)),ToTensor()]))
training_generator = data.DataLoader(dataset=training_set,batch_size=batch_size,shuffle=True, num_workers=number_of_workers)

validation_set = Dataset(val_ids, val_labels, 'val',transform=transforms.Compose([Rescale(512,512),ToTensor()]))
validation_generator = data.DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True, num_workers=number_of_workers)

test_set = Dataset(test_ids, test_labels, 'test', transform=transforms.Compose([Rescale(512.512),ToTensor()]))
test_generator = data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True,num_workers=number_of_workers)


model = vgg.create_vgg16()
# torch.nn.DataParallel(model.features)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(training_generator)






import torch
import numpy as np
from PIL import Image
from torch.utils import data
import os


class Dataset(data.Dataset):
    def __init__(self, folder, list_IDs, labels, type, transform=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.type = type
        self.transform = transform
        self.folder = folder

    def __len__(self):
       return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        file_image = os.path.join(self.folder, ID)
        x = Image.open(file_image)
        y = self.labels[ID]
        if self.transform:
            x = self.transform(x)
        return x, y








#
# class Rescale(object):
#     """Rescale the image in a sample to a given size.
#
#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image = sample['image']
#         lable = sample['label']
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         img = transform.resize(image, (new_h, new_w))
#
#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#
#         return {'image': img, 'label': lable}
#
#
# class RandomCrop(object):
#     """Crop randomly the image in a sample.
#
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h,
#                       left: left + new_w]
#         return {'image': image, 'label': label}
#
#
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         sample = {'image': torch.from_numpy(image),
#                 'label': torch.from_numpy(np.array(label))}
#         return sample


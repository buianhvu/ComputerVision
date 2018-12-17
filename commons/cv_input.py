# from commons.data_loading import *
from torch.utils import data
from commons.data_loading import Dataset

import os
import glob
from torchvision import transforms

DEFAULT_BATCH_SIZE = 40
DEFAULT_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
DEFAULT_WORKERS = 4


def get_ids_labels(ids_list: list, lab_dict: dict, path="/content/drive/My Drive/Food-11"):
    all_pictures = os.path.join(path, "*.jpg")
    for file in glob.glob(all_pictures):
        file_name = os.path.basename(file)
        name = file_name[:-4]
        label = name.split("_")[0]
        lab_dict[file_name] = int(label)
        ids_list.append(file_name)
    return ids_list, lab_dict


def get_loader(root="/content/drive/My Drive/Food-11", inside="training", target_type='train', batch_size=DEFAULT_BATCH_SIZE,
               input_transform=DEFAULT_TRANSFORM, number_of_workers=DEFAULT_WORKERS,
               shuffle=True):
    target_folder = os.path.join(root, inside)
    target_ids, target_labels = get_ids_labels([], {}, target_folder)
    if len(target_ids) == 0 or len(target_labels) == 0:
        return None, None
    target_set = Dataset(target_folder, target_ids, target_labels, target_type, transform=input_transform)
    target_loader = data.DataLoader(dataset=target_set, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=number_of_workers)
    return target_set, target_loader

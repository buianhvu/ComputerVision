from train.train import *
import os


if __name__ == '__main__':
    model_name = "vgg_food"
    output_folder = "output/" + model_name
    os.makedirs(output_folder, exist_ok=True)
    clear_path(model_name, output_folder, output_folder)
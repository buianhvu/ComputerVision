from train.train import *


if __name__ == '__main__':
    model_name = "se18_food"
    output_folder = "/content/drive/My Drive/output/" + model_name
    os.makedirs(output_folder, exist_ok=True)
    clear_path(model_name, output_folder, output_folder)
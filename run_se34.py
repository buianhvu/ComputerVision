from model.se_net import *
from train.train import *
from test.test import *
from commons.cv_input import *


if __name__ == '__main__':
    model_name = "se34_food"
    num_classes = 11
    input_folder = "/content/drive/My Drive/Food-11"
    output_folder = "/content/drive/My Drive/output/" + model_name
    batch_size = 50
    epoch_num = 50

    net = res_se_50(num_classes=num_classes)

    _, train_loader = get_loader(root=input_folder, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = train_model_ac_load_save(net, train_loader, model_name=model_name, optimize_func=OPT["adam"],
                                   momentum=None, path_state=output_folder,
                                   path_log=output_folder, epoch_num=epoch_num)

    _, test_loader = get_loader(root=input_folder, inside="evaluation", batch_size=batch_size)
    classes = tuple(range(num_classes))
    test_data(model=net, test_loader=test_loader, classes=classes, device=device, file_log=output_folder)



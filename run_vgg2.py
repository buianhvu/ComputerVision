from model.vgg import *
from train.train import *
from test.test import *
from commons.cv_input import *


if __name__ == '__main__':
    model_name = "res_food"
    num_classes = 11
    input_folder = "/content/drive/My Drive/Food-11"
    output_folder = "/content/drive/My Drive/output/" + model_name
    batch_size = 50
    epoch_num = 50

    net = set_vgg16(num_classes=11, defaul_last_size=(7, 7))

    _, train_loader = get_loader(root=input_folder, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = train_model_ac_load_save(net, train_loader, model_name=model_name, optimize_func=OPT["adam"],
                                   momentum=None, path_state=output_folder,
                                   path_log=output_folder, epoch_num=epoch_num)

    _, test_loader = get_loader(root=input_folder, inside="evaluation", batch_size=batch_size)
    classes = tuple(range(num_classes))
    test_data(model=net, test_loader=test_loader, classes=classes, device=device, file_log=output_folder)

    #
    # net = res_net101(num_classes=11)
    # _, train_loader = get_loader(root="/content/drive/My Drive/Food-11", batch_size=50)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # net = train_model_ac_load_save(net, train_loader, model_name="res_food", optimize_func=OPT["adam"],
    #                                momentum=None, path_state="/content/drive/My Drive/output/" + "res_food",
    #                                path_log="/content/drive/My Drive/output"+"res_food", epoch_num=50)
    #
    # _, test_loader = get_loader(root="/content/drive/My Drive/Food-11",inside="evaluation", batch_size=50)
    # classes = tuple(range(11))
    # test_data(model=net, test_loader=test_loader, classes=classes, device=device)
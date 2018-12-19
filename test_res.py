from model.res_net import *
from train.train import *
from test.test import *
from commons.cv_input import *


if __name__ == '__main__':
    net = res_net101(num_classes=11)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_data(net, model_name="res_101_food", device=device)

    _, test_loader = get_loader(inside="evaluation")
    classes = tuple(range(11))
    test_data(model=net, test_loader=test_loader, classes=classes, device=device)



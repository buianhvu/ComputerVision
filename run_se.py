from model.se_net import *
from train.train import *
from test.test import *
from commons.cv_input import *


if __name__ == '__main__':
    net = se_default()
    _, train_loader = get_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = train_model(net, train_loader, model_name="se_food", optimize_func=OPT["adam"],
                      momentum=None)

    _, test_loader = get_loader(inside="evaluation")
    classes = tuple(range(11))
    test_data(model=net, test_loader=test_loader, classes=classes, device=device)


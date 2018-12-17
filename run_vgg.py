from model.vgg import *
from train.train import *
from test.test import *
from commons.cv_input import *


if __name__ == '__main__':
    vgg_16 = set_vgg16(num_classes=11)
    _, train_loader = get_loader(batch_size=40)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg_16 = train_model(vgg_16, train_loader, model_name="vgg_16_food", optimize_func=OPT["adam"],
                         momentum=None, device=device)

    _, test_loader = get_loader(inside="evaluation")
    classes = tuple(range(11))
    test_data(model=vgg_16, test_loader=test_loader, classes=classes, device=device, batch_size=4)



from model.res_net import *
from train.train import *
from test.test import *
from commons.cv_input import *
from parser.parser import *

if __name__ == '__main__':
    args = parse_arguments()
    num_class = args.nclass
    output_dir = args.output
    log_dir = args.plog
    epoch = args.epoch
    opt_func = args.opt

    net = res_net101(num_classes=args.nclass)
    _, train_loader = get_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = train_model(net, train_loader, model_name="res_101_food", optimize_func=OPT["adam"],
                momentum=None)

    _, test_loader = get_loader(inside="evaluation")
    classes = tuple(range(args.nc))
    test_data(model=net, test_loader=test_loader, classes=classes, device=device)


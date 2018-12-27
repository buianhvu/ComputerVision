from train.train import *
from test.test import *
from commons.cv_input import *
from parser.parser import *
import os


def training_testing(net_func):
    args = parse_arguments()
    input_dir = args.input
    output_dir = args.output
    path_log = args.plog
    epoch = args.epoch
    opt_func = args.opt_func
    learning_rate = args.lr
    momentum = args.momen
    model_name = args.model_name
    init = args.init
    num_class = args.nclass
    batch_size = args.batch_size

    os.makedirs(output_dir, exist_ok=True)
    net = net_func(num_classes=args.nclass)  # call create net function
    _, train_loader = get_loader(root=input_dir, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt_func == 'adam':
        momentum = None
    net = train_model_ac_load_save(net, train_loader, model_name=model_name, path_state=output_dir,
                                   path_log=path_log, learning_rate=learning_rate, momentum=momentum,
                                   optimize_func=OPT[opt_func], epoch_num=epoch, device=device)

    _, test_loader = get_loader(root=input_dir, inside="evaluation", batch_size=batch_size)
    classes = tuple(range(num_class))
    test_data(model=net, test_loader=test_loader, classes=classes, device=device)


def training(net_func):
    args = parse_arguments()
    input_dir = args.input
    output_dir = args.output
    path_log = args.plog
    epoch = args.epoch
    opt_func = args.opt_func
    learning_rate = args.lr
    momentum = args.momen
    model_name = args.model_name
    init = args.init
    num_class = args.nclass
    batch_size = args.batch_size

    os.makedirs(output_dir, exist_ok=True)
    net = net_func(num_classes=args.nclass)  # call create net function
    _, train_loader = get_loader(root=input_dir, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt_func == 'adam':
        momentum = None
    net = train_model(net, train_loader, model_name=model_name, path_state=output_dir,
                      path_log=path_log, learning_rate=learning_rate, momentum=momentum,
                      optimize_func=OPT[opt_func], epoch_num=epoch, device=device)



from commons.main_func import *
from model.res_net import *

if __name__ == '__main__':
    training_testing(res_net101)
    # args = parse_arguments()
    # input_dir = args.input
    # output_dir = args.output
    # path_log = args.plog
    # epoch = args.epoch
    # opt_func = args.opt_func
    # learning_rate = args.lr
    # momentum = args.momen
    # model_name = args.model_name
    # init = args.init
    # num_class = args.nclass
    # batch_size = args.batch_size
    #
    # os.makedirs(output_dir, exist_ok=True)
    # net = res_net101(num_classes=args.nclass)
    # _, train_loader = get_loader(root=input_dir, batch_size=batch_size)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # if opt_func == 'adam':
    #     momentum = None
    # net = train_model(net, train_loader, model_name=model_name, path_state=output_dir,
    #                   path_log=path_log, learning_rate=learning_rate, momentum=momentum,
    #                   optimize_func=OPT[opt_func], epoch_num=epoch, device=device)
    #
    # _, test_loader = get_loader(root=input_dir, inside="evaluation", batch_size=batch_size)
    # classes = tuple(range(num_class))
    # test_data(model=net, test_loader=test_loader, classes=classes, device=device)



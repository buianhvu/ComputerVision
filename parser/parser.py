from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', default='output/default_net/')
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--opt-func', default="adam", type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momen', default=0.9, type=float)
    parser.add_argument('--model-name', default='se_net', type=str)
    parser.add_argument('--init', default=True, type=bool)
    return parser.parse_args()


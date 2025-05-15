import argparse
import sys

args = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Robust Repair",
                                     epilog="End of Parameters")

    # Parameters for training
    parser.add_argument(
        "--mode", help="train models and probes, repair model, or test models", type=str, default='train',
    )
    parser.add_argument(
        "--gpu", help="use which gpu", type=int, default=0,
    )
    parser.add_argument(
        "--set", help="use which dataset", type=str, default="IMAGENET10",
    )
    parser.add_argument(
        "--arch", help="use which model", type=str, default="stdvgg16_class10",
    )
    parser.add_argument(
        "--bdtype", help="use which bd method", type=str, default="BadNets",
    )
    parser.add_argument(
        "--pretrain", help="whether use pretrain parameter for backdoor model", type=bool, default=False,
    )
    parser.add_argument(
        "--train", help="whether train backdoor model", type=bool, default=False,
    )
    parser.add_argument(
        "--saveRes", help="whether save backdoor model", type=bool, default=False,
    )
    args = parser.parse_args()

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
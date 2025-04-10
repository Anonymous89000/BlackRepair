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

    args = parser.parse_args()

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
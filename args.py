import argparse
import sys

args = None
def str_to_bool(value):
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

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
        "--adtype", help="use which ad method", type=str, default="PGD",
    )
    parser.add_argument(
        "--pretrain", help="whether use pretrain parameter for backdoor model", type=str_to_bool, default=False,
    )
    parser.add_argument(
        "--train", help="whether train backdoor model", type=str_to_bool, default=False,
    )
    parser.add_argument(
        "--saveRes", help="whether save backdoor model", type=str_to_bool, default=False,
    )
    parser.add_argument(
        "--pretrainfile", help="where to load pretrained model", type=str, default=None,
    )
    parser.add_argument(
        "--includewrong", help="whether ", type=str_to_bool, default=False,
    )
    parser.add_argument(
        "--advdataset", help="adversarial dataset ", type=str, default=None,
    )

    args = parser.parse_args()

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
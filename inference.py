import os

import torch

from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    args.data_dir = os.environ.get("SM_CHANNEL_EVAL", args.data_dir)
    args.model_dir = os.environ.get("SM_CHANNEL_MODEL", args.model_dir)
    args.output_dir = os.environ.get("SM_OUTPUT_DATA_DIR ", args.output_dir)
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

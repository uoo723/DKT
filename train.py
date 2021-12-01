import os
from datetime import datetime

import torch
import wandb
from logzero import logger

import dkt.trainer as trainer
from args import parse_args
from dkt.dataloader import Preprocess, partition_question
from dkt.utils import log_elapsed_time, set_logger, setSeeds


def print_args(args):
    msg = "\n"
    for k, v in vars(args).items():
        msg += f"{k}: {v}\n"
    logger.info(msg)


@log_elapsed_time
def main(args):
    # wandb.login()

    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    args.data_dir = os.environ.get("SM_CHANNEL_TRAIN", args.data_dir)
    args.model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)

    if args.output_root_dir:
        args.output_dir = args.output_root_dir
    else:
        args.output_dir = os.path.join(
            args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    args.model_dir = os.path.join(args.output_dir, "model")

    set_logger(os.path.join(args.output_dir, "train.log"))
    print_args(args)

    if not args.inference_only:
        preprocess = Preprocess(args)
        preprocess.load_train_data(args.file_name)
        data = preprocess.get_train_data()
        try:
            trainer.run(args, data)
        except KeyboardInterrupt:
            logger.info("Stop training")

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

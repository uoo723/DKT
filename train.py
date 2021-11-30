import os
from datetime import datetime
import torch
import wandb

from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess, partition_question
from dkt.utils import set_logger, setSeeds
from logzero import logger


def print_args(args):
    msg = "\n"
    for k, v in vars(args).items():
        msg += f"{k}: {v}\n"
    logger.info(msg)


def main(args):
    # wandb.login()

    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    args.data_dir = os.environ.get("SM_CHANNEL_TRAIN", args.data_dir)
    args.model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)

    args.output_dir = os.path.join(
        args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    args.model_dir = os.path.join(args.output_dir, "model")

    set_logger(os.path.join(args.output_dir, "train.log"))
    print_args(args)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    data = preprocess.get_train_data()

    # train_data, valid_data = preprocess.split_data(train_data)

    # if args.partition_question:
    #     train_data = partition_question(train_data, args)

    # wandb.init(project="dkt", config=vars(args))
    # trainer.run(args, train_data, valid_data)
    trainer.run(args, data)

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    trainer.inference(args, test_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

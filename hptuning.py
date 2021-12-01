import copy
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict

import click
import numpy as np
import optuna
import torch
from attrdict import AttrDict
from logzero import logger
from optuna import Trial
from optuna.study.study import Study
from ruamel.yaml import YAML

import dkt.trainer as trainer
from dkt.dataloader import Preprocess
from dkt.utils import log_elapsed_time, set_logger, setSeeds


def get_hp_params(trial: Trial, hp_params: Dict):
    p = {}
    for key, value in hp_params.items():
        if value["type"] == "categorical":
            p[key] = trial.suggest_categorical(key, value["value"])
        elif value["type"] == "float":
            p[key] = trial.suggest_float(
                key, *value["value"], step=value.get("step", None)
            )
        elif value["type"] == "int":
            p[key] = trial.suggest_int(key, *value["value"])
    return p


def objective(trial: Trial, params: Dict, hp_params: Dict, data: np.ndarray):
    p = copy.deepcopy(params)
    p.update(get_hp_params(trial, hp_params))
    return trainer.run(p, data)


@click.command(context_settings={"show_default": True})
@click.option(
    "--model",
    type=click.Choice(["lstm", "sakt", "saint", "akt"]),
    default="sakt",
    help="model",
)
@click.option("--output-root-dir", type=click.Path(), help="output root directory")
@click.option("--output-dir", type=click.Path(exists=True), help="Set output directory")
@click.option(
    "--config-file-path",
    type=click.Path(exists=True),
    default="config/hp_params.yaml",
    help="hp params config file path",
)
@click.option(
    "--default-param-file-path",
    type=click.Path(exists=True),
    default="config/default_args.json",
    help="default param file path",
)
@click.option("--seed", type=click.INT, default=42, help="seed")
@click.option("--n-trials", type=click.INT, default=20, help="# of trials")
@click.option("--study-name", type=click.STRING, default="study", help="Set study name")
@click.option(
    "--storage-name",
    type=click.STRING,
    default="storage",
    help="Set storage name to save study",
)
@click.option("--n-epochs", type=click.INT, default=100, help="# of epochs")
@click.option("--patience", type=click.INT, default=8, help="early stop patience")
@click.option("--clip-grad", type=click.FLOAT, default=5.0, help="clipping value")
@click.option("--lr", type=click.FLOAT, default=1e-3, help="learning rate")
@click.option("--batch-size", type=click.INT, default=128, help="batch size")
@click.option("--model-name", type=click.STRING, default="model.pt", help="model name")
@click.option(
    "--logfile-name", type=click.STRING, default="train.log", help="logfile name"
)
@log_elapsed_time
def main(**args):
    args = AttrDict(args)

    yaml = YAML(typ="safe")
    hp_params = AttrDict(yaml.load(Path(args.config_file_path)))

    with open(args.default_param_file_path, "r", encoding="utf-8") as f:
        params = AttrDict(json.load(f))

    if args.output_dir:
        params.output_dir = args.output_dir
    else:
        if args.output_root_dir:
            params.output_dir = args.output_root_dir

        params.output_dir = os.path.join(
            params.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_hptuning"
        )

    params.device = "cuda" if torch.cuda.is_available() else "cpu"
    params.model_dir = os.path.join(params.output_dir, "model")
    params.model_name = args.model_name
    params.seed = args.seed
    params.model = args.model
    params.patience = args.patience
    params.clip_grad = args.clip_grad
    params.batch_size = args.batch_size
    params.n_epochs = args.n_epochs
    params.lr = args.lr
    params.k_folds = 1

    setSeeds(args.seed)

    logfile_path = os.path.join(params.output_dir, args.logfile_name)
    set_logger(logfile_path)
    optuna.logging.get_logger("optuna").addHandler(logger.handlers[-1])

    preprocess = Preprocess(params)
    preprocess.load_train_data(params.file_name)
    data = preprocess.get_train_data()

    storage_name = f"{args.storage_name}.db"
    study: Study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{os.path.join(params.output_dir, storage_name)}",
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(
            partial(objective, params=params, hp_params=hp_params, data=data),
            n_trials=args.n_trials,
        )
    except KeyboardInterrupt:
        logger.info("Stop tuning.")

    all_trials = sorted(
        study.trials, key=lambda x: x.value if x.value else 0, reverse=True
    )
    best_trial = all_trials[0]

    best_exp_num = best_trial.number
    best_score = best_trial.value
    best_params = best_trial.params

    logger.info(f"best_exp_num: {best_exp_num}")
    logger.info(f"best_score: {best_score}")
    logger.info(f"best_params:\n{best_params}")


if __name__ == "__main__":
    main()

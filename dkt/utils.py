import os
import random
from typing import Dict, Union

import logzero
import numpy as np
import torch
import torch.nn as nn
from logzero import logger

TModel = Union[nn.DataParallel, nn.Module]


def setSeeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def swa_init(model: TModel, swa_state: Dict[str, torch.Tensor]) -> None:
    logger.info("SWA Initializing")
    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] = 1
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()


def swa_step(model: TModel, swa_state: Dict[str, torch.Tensor]) -> None:
    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] += 1
    beta = 1.0 / swa_state["models_num"]
    with torch.no_grad():
        for n, p in model.named_parameters():
            swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)


def swap_swa_params(model: TModel, swa_state: Dict[str, torch.Tensor]) -> None:
    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    for n, p in model.named_parameters():
        p.data, swa_state[n] = swa_state[n], p.data


def set_logger(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logzero.logfile(log_path)

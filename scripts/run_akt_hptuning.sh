#!/usr/bin/env bash

args=(
    --seed 20417
    --model 'akt'
    --output-root-dir '/output'
    --config-file-path 'config/akt_hp_params.yaml'
    --n-trials 40
    --n-epochs 200
    --output-dir '/output/20211130_205807_hptuning'
    --logfile-name 'train.log'
    --model-name 'model.pt'
)

python hptuning.py "${args[@]}"

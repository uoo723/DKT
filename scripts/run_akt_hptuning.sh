#!/usr/bin/env bash

args=(
    --seed 20417
    --model 'akt'
    --n-trials 20
    --config-file-path 'config/akt_hp_params.yaml'
)

python hptuning.py "${args[@]}"

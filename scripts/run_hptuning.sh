#!/usr/bin/env bash

args=(
    --seed 2048
    --model 'sakt'
    --n-trials 20
)

python hptuning.py "${args[@]}"

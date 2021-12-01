#!/usr/bin/env bash

args=(
    --seed 2070
    --model 'sakt'
    --n-trials 20
)

python hptuning.py "${args[@]}"

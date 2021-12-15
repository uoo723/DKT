#!/usr/bin/env bash

args=(
    --seed 44818
    --n_epochs 100
    --batch_size 128
    --max_seq_len 20
    --hidden_dim 64
    --n_heads 4
    --n_layers 4
    --patience 8
    --lr 1e-3
    --drop_out 0.4
    --model 'sakt'
    --clip_grad 5.0
    --partition_question
    --interaction_type 2
    --k_folds 5
    --enable_da
)

python train.py "${args[@]}"

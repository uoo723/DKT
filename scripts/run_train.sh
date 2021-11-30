#!/usr/bin/env bash

args=(
    --seed 2021
    --n_epochs 1
    --batch_size 256
    --max_seq_len 50
    --hidden_dim 64
    # --final_fc_dim 128
    # --ff_dim 1024
    --n_heads 4
    --n_layers 4
    --patience 8
    --lr 1e-3
    --drop_out 0.4
    --model 'sakt'
    --clip_grad 5.0
    --partition_question
    --interaction_type 0
    --k_folds 5
    --enable_da
    # --random_permute
    # --attn_direction bi
    --model_name 'model.pt'
    --output_filename 'output.csv'
)

python train.py "${args[@]}"

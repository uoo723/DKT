#!/usr/bin/env bash

args=(
    --seed 2021
    --n_epochs 60
    --batch_size 128
    --max_seq_len 100
    --hidden_dim 128
    --final_fc_dim 64
    --ff_dim 256
    --n_heads 4
    --n_layers 4
    --patience 8
    --lr 1e-3
    --drop_out 0.4
    --model 'akt'
    --clip_grad 5.0
    --partition_question
    --interaction_type 0
    --k_folds 5
    # --scheduler 'linear_warmup'
    # --random_permute
    # --attn_direction bi
    --enable_da
    --model_name 'model_akt.pt'
    --output_filename 'output_akt.csv'
)

python train.py "${args[@]}" && \
python inference.py "${args[@]}"

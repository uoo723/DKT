#!/usr/bin/env bash

args=(
    --seed 2021
    --n_epochs 100
    --batch_size 256
    --max_seq_len 50
    --hidden_dim 128
    --final_fc_dim 64
    --ff_dim 256
    --n_heads 4
    --n_layers 2
    --patience 8
    --lr 1e-3
    --drop_out 0.4
    --model 'saint'
    --clip_grad 5.0
    --partition_question
    --interaction_type 0
    --k_folds 5
    # --random_permute
    # --attn_direction bi
    --model_name 'model_saint.pt'
    --output_filename 'output_saint.csv'
)

python train.py "${args[@]}" && \
python inference.py "${args[@]}"

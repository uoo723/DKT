#!/usr/bin/env bash

args=(
    --seed 2021
    --n_epochs 120
    --batch_size 256
    --max_seq_len 30
    --hidden_dim 64
    # --final_fc_dim 128
    # --ff_dim 1024
    --n_heads 4
    --n_layers 4
    --patience 8
    --lr 1e-4
    --drop_out 0.4
    --model saint
    --clip_grad 5.0
    --partition_question
    --interaction_type 0
    --k_folds 5
    # --random_permute
    # --attn_direction bi
    --model_name 'model3.pt'
    --output_filename 'output3.csv'
)

python train.py "${args[@]}" && \
python inference.py "${args[@]}"

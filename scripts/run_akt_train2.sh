#!/usr/bin/env bash

args=(
    --seed 20418
    --n_epochs 200
    --batch_size 128
    --max_seq_len 200
    --hidden_dim 128
    --final_fc_dim 512
    --ff_dim 128
    --n_heads 4
    --n_layers 4
    --patience 8
    --lr 1e-3
    --drop_out 0.2
    --model 'akt'
    --clip_grad 5.0
    --partition_question
    # --enable_da
    --interaction_type 2
    --k_folds 5
    # --inference_only
    # --output_root_dir 'output/20211201_091700'
    # --enable_da
    # --random_permute
    # --attn_direction bi
    # --model_name 'model.pt'
    # --output_filename 'output.csv'
)

python train.py "${args[@]}"

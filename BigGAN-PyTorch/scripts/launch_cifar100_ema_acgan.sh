#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 2000 --save_every 2000 --num_best_copies 1 --num_save_copies 0 --seed 0 \
--loss acgan --G_lambda 0.2 --D_lambda 0.2 --experiment_name c100_acgan_n4_b50_e1k_w02

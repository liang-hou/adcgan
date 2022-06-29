#!/bin/bash
python train.py \
--dataset TI200 --parallel --shuffle --augment --batch_size 50 \
--num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 500 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 64 --D_ch 64 \
--ema --use_ema --ema_start 20000 \
--test_every 1000 --save_every 1000 --num_best_copies 1 --num_save_copies 1 --seed 0 \
--loss adcgan --G_lambda 0.5 --D_lambda 0.5 --experiment_name ti200_adcgan_n2_b100_e500_a20000_w05

#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --output_dir ./out/wikibio/bart_few100/model \
        --train_file ./data/wikibio/train_100 \
        --train_file_unlabel ./data/wikibio/unlabel \
        --predict_file ./data/wikibio/val \
        --substitution_file ./data/counterfitted_neighbors.json \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --curriculum [2,4,10] \
        --train_batch_size 24 \
        --train_batch_size_unlabel 24 \
        --predict_batch_size 32 \
        --predict_batch_size_unlabel 32 \
        --max_input_length 384 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --num_train_epochs_unlabel 1 \
        --warmup_steps 8 \
        --warmup_steps_unlabel 400 \
        --eval_period 8 \
        --num_beams 5 \
        --struct_noise 0.4 \
        --semantic_noise 0.4 \
        --cover_ratio 0.2 \
        --seed 42


CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --output_dir ./out/wikibio/bart_few50/model \
        --train_file ./data/wikibio/train_50 \
        --train_file_unlabel ./data/wikibio/unlabel \
        --predict_file ./data/wikibio/val \
        --substitution_file ./data/counterfitted_neighbors.json \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --curriculum [2,4,10] \
        --train_batch_size 24 \
        --train_batch_size_unlabel 24 \
        --predict_batch_size 32 \
        --predict_batch_size_unlabel 32 \
        --max_input_length 384 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --num_train_epochs_unlabel 1 \
        --warmup_steps 4 \
        --warmup_steps_unlabel 400 \
        --eval_period 4 \
        --num_beams 5 \
        --struct_noise 0.4 \
        --semantic_noise 0.4 \
        --cover_ratio 0.2 \
        --seed 42


CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --output_dir ./out/wikibio/bart_few200/model \
        --train_file ./data/wikibio/train_200 \
        --train_file_unlabel ./data/wikibio/unlabel \
        --predict_file ./data/wikibio/val \
        --substitution_file ./data/counterfitted_neighbors.json \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --curriculum [2,4,10] \
        --train_batch_size 24 \
        --train_batch_size_unlabel 24 \
        --predict_batch_size 32 \
        --predict_batch_size_unlabel 32 \
        --max_input_length 384 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --num_train_epochs_unlabel 1 \
        --warmup_steps 16 \
        --warmup_steps_unlabel 400 \
        --eval_period 16 \
        --num_beams 5 \
        --struct_noise 0.4 \
        --semantic_noise 0.4 \
        --cover_ratio 0.2 \
        --seed 42 


CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --output_dir ./out/wikibio/bart_few500/model \
        --train_file ./data/wikibio/train_500 \
        --train_file_unlabel ./data/wikibio/unlabel \
        --predict_file ./data/wikibio/val \
        --substitution_file ./data/counterfitted_neighbors.json \
        --model_path ./bart_model \
        --tokenizer_path ./bart_model \
        --curriculum [2,4,10] \
        --train_batch_size 24 \
        --train_batch_size_unlabel 24 \
        --predict_batch_size 32 \
        --predict_batch_size_unlabel 32 \
        --max_input_length 384 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --num_train_epochs_unlabel 1 \
        --warmup_steps 40 \
        --warmup_steps_unlabel 400 \
        --eval_period 40 \
        --num_beams 5 \
        --struct_noise 0.4 \
        --semantic_noise 0.4 \
        --cover_ratio 0.2 \
        --seed 42

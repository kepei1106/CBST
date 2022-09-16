from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch

from run_gt import self_training


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--train_file", default="data/nqopen-train.json")
    parser.add_argument("--train_file_unlabel", default="data/nqopen-train.json")
    parser.add_argument("--predict_file", default="data/nqopen-dev.json")
    parser.add_argument("--substitution_file", default="./data/counterfitted_neighbors.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')

    # Model parameters
    parser.add_argument("--model_name", type=str, default="bart")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--model_path", type=str, default="./bart_model")
    parser.add_argument("--tokenizer_path", type=str, default="./bart_model")

    # Preprocessing/decoding parameters
    parser.add_argument('--max_input_length', type=int, default=32)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument("--append_another_bos", action='store_true', default=False)
    parser.add_argument("--remove_bos", action='store_true', default=False)
    parser.add_argument("--clean_up_spaces", action='store_true', default=False)

    # Training parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size_unlabel", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--predict_batch_size_unlabel", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps_unlabel", default=4, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs_unlabel", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps_unlabel", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=15)
    parser.add_argument("--no_lr_decay", action='store_true', default=False)
    parser.add_argument('--iter_st', type=int, default=1)
    parser.add_argument('--curriculum_type', type=str, default='triple',
                        help="triple or length")
    parser.add_argument('--curriculum', type=str, default='[2, 6, 10]',
                        help="Split of curriculum")
    parser.add_argument("--ppl_ratio", default=0.5, type=float,
                        help="Ratio for selection of PPL in self-training")
    parser.add_argument("--cover_ratio", default=1.0, type=float,
                        help="Ratio for selection of coverage in self-training")
    parser.add_argument("--struct_noise", default=0.0, type=float,
                        help="structure noise in pseudo-training")
    parser.add_argument("--semantic_noise", default=0.0, type=float,
                        help="semantic noise in pseudo-training")

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate & save model")
    parser.add_argument('--eval_period_unlabel', type=int, default=1000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Number of workers for dataloaders")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs
    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))
    self_training(args, logger)


if __name__=='__main__':
    main()
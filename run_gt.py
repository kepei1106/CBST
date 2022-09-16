import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from data import GTDataset, GTDataLoader, evaluate_bleu, GTUnlabelDataLoader
from bart import MyBart
from t5 import MyT5
from tqdm import tqdm, trange
import random
import json


def self_training(args, logger):
    # Load tokenizer
    if args.model_name == "bart":
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)

    # Load train / dev data
    train_data = GTDataset(logger, args, args.train_file, tokenizer, "train")
    dev_data = GTDataset(logger, args, args.predict_file, tokenizer, "val")
    train_dataloader = GTDataLoader(args, train_data, "train")
    dev_dataloader = GTDataLoader(args, dev_data, "val")

    # First of all, train the model on labeled data
    run(args, logger, args.model_path, tokenizer, train_dataloader, dev_dataloader, 0)

    if args.do_train:
        # self-training process
        unlabel_data = GTDataset(logger, args, args.train_file_unlabel, tokenizer, "train")
        for c_id in range(len(eval(args.curriculum))):
            # collect unlabeled data satisfying the current curriculum
            unlabel_data.build_data_from_cur()
            unlabel_dev_dataloader = GTUnlabelDataLoader(args, unlabel_data, "val")
            # generate pseudo-labeled data with the teacher model
            pseudo_label, pred_prob = inference_for_genwiki(args, logger, args.output_dir, tokenizer, unlabel_dev_dataloader, c_id)
            print('number of unlabeled data: ', len(pseudo_label))
            unlabel_data.update(pseudo_label, pred_prob)

            # output pseudo-labeled data
            save_path_tmp = os.path.join(args.output_dir, "{}pseudo_data_{}.json".format(args.prefix, c_id))
            with open(save_path_tmp, 'w', encoding='utf8') as f_out:
                json.dump(unlabel_data.data, f_out, sort_keys=True, indent=4)
            print('number of pseudo data: ', len(unlabel_data.data))

            # pretrain on pseudo-labeled data (with noise), finetune on labeled data (without noise)
            unlabel_data.set_noise(True)
            unlabel_train_dataloader = GTUnlabelDataLoader(args, unlabel_data, "train")
            run_unlabel(args, logger, args.model_path, tokenizer, unlabel_train_dataloader, c_id)
            unlabel_data.set_noise(False)
            run(args, logger, args.output_dir, tokenizer, train_dataloader, dev_dataloader, c_id)
            unlabel_data.curriculum_next()


def run_unlabel(args, logger, model_path, tokenizer, train_dataloader, c_id):
    # Prepare the training on pseudo-labeled data
    model = MyBart.from_pretrained(model_path) if args.model_name == "bart" \
            else MyT5.from_pretrained(model_path)
    if args.n_gpu>1:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    t_total = len(train_dataloader) // args.gradient_accumulation_steps_unlabel * args.num_train_epochs_unlabel
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.no_lr_decay:
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps_unlabel,
                                        num_training_steps=t_total)
    else:
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=1000000)
    train_unlabel(args, logger, model, train_dataloader, optimizer, scheduler, tokenizer, c_id)


def inference_for_genwiki(args, logger, model_path, tokenizer, dev_dataloader, c_id):
    # Generate pseudo labels for unlabeled data
    checkpoint = model_path + '_label_' + str(c_id)
    model = MyBart.from_pretrained(checkpoint) if args.model_name == "bart" \
            else MyT5.from_pretrained(checkpoint)
    logger.info("Loading checkpoint from {}".format(checkpoint))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()
    predictions = []
    predictions_scores = []

    for i, batch in enumerate(dev_dataloader):
        if i % 100 == 0:
            print('inference genwiki: ', i)
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        # outputs: output token ids
        # outputs_scores: log generation probability of each sample
        outputs, outputs_scores = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=args.num_beams,
                                 length_penalty=args.length_penalty,
                                 max_length=args.max_output_length,
                                 early_stopping=True,)
        for output_id, output in enumerate(outputs):
            # transfer subword ids into strings
            pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            predictions.append(pred.strip())
            predictions_scores.append(outputs_scores[output_id].cpu().item())

    # save pseudo-labeled data
    save_path = os.path.join(args.output_dir, "{}pseudo_data_{}.txt".format(args.prefix, c_id))
    with open(save_path, "w") as f:
        for pred_id in range(len(predictions)):
            f.write(predictions[pred_id] + '\t' + str(predictions_scores[pred_id]) + '\n')
    logger.info("Saved pseudo data in {}".format(save_path))

    return predictions, predictions_scores


def run(args, logger, model_path, tokenizer, train_dataloader, dev_dataloader, c_id):
    # Prepare the training on the labeled data
    if args.do_train:
        if model_path == args.model_path:
            model = MyBart.from_pretrained(model_path) if args.model_name == "bart" \
                else MyT5.from_pretrained(model_path)
        else:
            checkpoint = model_path + '_unlabel_' + str(c_id)
            model = MyBart.from_pretrained(checkpoint) if args.model_name == "bart" \
                else MyT5.from_pretrained(checkpoint)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if not args.no_lr_decay:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=t_total)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=1000000)
        if model_path == args.model_path:
            train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer, c_id)
        else:
            train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer, c_id + 1)

    # inference mode
    if args.do_predict:
        checkpoint = args.output_dir
        model = MyBart.from_pretrained(checkpoint) if args.model_name == "bart" \
            else MyT5.from_pretrained(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=True)
        logger.info("%s on %s data: %.4f" % (dev_dataloader.dataset.metric, dev_dataloader.dataset.data_type, ems))


def train_unlabel(args, logger, model, train_dataloader, optimizer, scheduler, tokenizer, c_id):
    # Train on the pseudo-labeled data
    model.train()
    global_step = 0
    train_losses = []

    train_iterator = trange(int(args.num_train_epochs_unlabel), desc="Epoch")
    logger.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps_unlabel == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Only output the log information without validation
            if global_step % args.eval_period_unlabel == 0:
                logger.info("Step %d Train loss %.2f Learning rate %.2e on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    scheduler.get_lr()[0],
                    epoch))
                train_losses = []

        # Save the model
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir + '_unlabel_' + str(c_id))
        logger.info("Saving model on epoch=%d, global_step=%d" % (epoch, global_step))


def train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer, c_id):
    # Train on the labeled data
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Run the model on the validation set
            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(model if args.n_gpu == 1 else model.module, dev_dataloader, tokenizer, args, logger)
                logger.info("Step %d Train loss %.2f Learning rate %.2e %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    scheduler.get_lr()[0],
                    dev_dataloader.dataset.metric,
                    curr_em * 100,
                    epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir + '_label_' + str(c_id))
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (dev_dataloader.dataset.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()

        if stop_training:
            break


def inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=False):
    # Inference on the valid / test set
    predictions = []
    for i, batch in enumerate(dev_dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        # outputs_all: output token ids
        outputs_all = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=args.num_beams,
                                 length_penalty = args.length_penalty,
                                 max_length=args.max_output_length,
                                 early_stopping=True,)
        if args.num_beams == 1:
            outputs = outputs_all
        else:
            outputs, outputs_scores = outputs_all[0], outputs_all[1]
        for output_id, output in enumerate(outputs):
            # transfer subword ids into strings
            pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            predictions.append(pred.strip())
    if save_predictions:
        save_path = os.path.join(args.output_dir, "{}predictions.txt".format(args.prefix))
        with open(save_path, "w") as f:
            for pred_id in range(len(predictions)):
                f.write(predictions[pred_id] + '\n')
        logger.info("Saved prediction in {}".format(save_path))
    data_ref = [data_ele['text'] for data_ele in dev_dataloader.dataset.data]
    assert len(predictions) == len(data_ref)
    # Compute the metrics
    return evaluate_bleu(data_ref=data_ref, data_sys=predictions)

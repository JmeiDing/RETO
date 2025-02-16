import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append('/Users/dingjunmei/code/RL_LLM')


import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import pandas as pd
import csv
import time

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric, Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from bert_score import BERTScorer

from sft4lms.Summarization.data_loader import get_data_split, get_general_data_split, get_attack_data_split, EXTRACTION_PREFIX, SUMMARIZATION_PREFIX, SPLIT
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords

# 在每个训练迭代结束后手动清理 CUDA 缓存
torch.cuda.empty_cache()
# check_min_version("4.35.0.dev0")
require_version("datasets>=1.8.0")
logger = get_logger(__name__)


def load_model(output_dir, model_path, strategy):
    if output_dir and os.path.exists(output_dir):
        if "checkpoint" in output_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
        else:
            if strategy == 'load_last':
                latest_checkpoint_idx = 0
                dir_list = os.listdir(output_dir) # find the latest checkpoint
                for d in dir_list:
                    if "checkpoint" in d and "best" not in d:
                        checkpoint_idx = int(d.split("-")[-1])
                        if checkpoint_idx > latest_checkpoint_idx:
                            latest_checkpoint_idx = checkpoint_idx
                if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")):
                    ft_model_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                    return model, tokenizer
            elif strategy == 'load_best':
                ft_model_path = os.path.join(output_dir, f"best_checkpoint")
                if os.path.exists(ft_model_path):
                    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                    return model, tokenizer
            elif strategy == 'load_initial':
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer

    # load pretrained model for hf
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def fine_tune_hf(
    task,
    model_name,
    dataset_name,
    n_train,
    push_to_hub,
    model,
    tokenizer,
    extraction_source,
    max_ctx_len,
    max_tgt_len,
    output_dir,
    train_data,
    val_data,
    test_data,
    pad_to_max_length,
    epochs,
    train_batch_size,
    eval_batch_size,
    learning_rate,
    lr_scheduler_type,
    seed,
    num_warmup_steps,
    gradient_accumulation_steps,
    max_train_steps,
    checkpointing_steps,
    ignore_pad_token_for_loss,
    with_tracking,
    report_to,
    weight_decay
):

    accelerator_log_kwargs = {}

    if with_tracking:
        accelerator_log_kwargs["log_with"] = report_to
        accelerator_log_kwargs["project_dir"] = output_dir

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, **accelerator_log_kwargs)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    def preprocess_function_for_summarization(batch):
        inputs = [SUMMARIZATION_PREFIX + doc for doc in batch["article"]]
        targets = [doc for doc in batch["summary"]]
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_for_extraction(batch):
        inputs = [EXTRACTION_PREFIX + doc for doc in batch["article"]]
        targets = [doc for doc in batch["target"]]
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        for target in targets:
            print('target:',target)
        # 计算每个样本target的数量
        target_element_counts = np.array([len(row) for row in targets])
        # 统计每个样本编码后labels的数量频次
        target_count_counts = np.bincount(target_element_counts)
        for count, frequency in enumerate(target_count_counts):
            if frequency > 0:
                print(f"包含 {count} 个元素targets的数量：{frequency}")

        # 计算每个样本编码后labels的数量
        label_element_counts = np.array([len(row) for row in model_inputs["labels"]])
        # 统计每个样本编码后labels的数量频次
        label_count_counts = np.bincount(label_element_counts)
        for count, frequency in enumerate(label_count_counts):
            if frequency > 0:
                print(f"包含 {count} 个元素labels的数量：{frequency}")

        return model_inputs

    def convert_label_to_key(labels):
        return " ".join([str(i) for i in labels])
    
    def dataset_summary_mapping(dataset):
        mapping_dict = {}
        for d in dataset:
            label = d['labels']
            label_key = convert_label_to_key(label)
            mapping_dict[label_key] = d['summary']
            #print(mapping_dict)
        return mapping_dict

    #{"article": article, "summary": summary, "id": id, "phrases": selected_phrases, "target": target}
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)

    train_dataset = train_dataset.map(remove_columns=["phrases"])
    val_dataset = val_dataset.map(remove_columns=["phrases"])
    test_dataset = test_dataset.map(remove_columns=["phrases"])
    # tokenize the dataset
    if task == 'summarization':
        train_dataset = train_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["id", "target", "article", "summary"])
        val_dataset = val_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["id", "target", "article"])
        test_dataset = test_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["id", "target", "article"])
    elif task == 'extraction':
        train_dataset = train_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["id","target", "article", "summary"])
        val_dataset = val_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["id", "target", "article"])
        test_dataset = test_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["id", "target", "article"])

    # print('train_dataset:',train_dataset)
    # train_dataset: Dataset({
    #     features: ['input_ids', 'attention_mask', 'labels'],
    #     num_rows: 10
    # })
    # print('val_dataset:', val_dataset)
    # val_dataset: Dataset({
    #     features: ['summary', 'input_ids', 'attention_mask', 'labels'],
    #     num_rows: 5
    # })

    # mapping from label to summaries
    val_summary_mapping = dataset_summary_mapping(val_dataset)
    test_summary_mapping = dataset_summary_mapping(test_dataset)
    val_test_summary_mapping = {**val_summary_mapping, **test_summary_mapping}
    val_dataset = val_dataset.map(remove_columns=["summary"])
    test_dataset = test_dataset.map(remove_columns=["summary"])
    # print('val_dataset', val_dataset)
    # val_dataset
    # Dataset({
    #     features: ['input_ids', 'attention_mask', 'labels'],
    #     num_rows: 5
    # })

    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.mixed_precision == 'fp16' else None,
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,batch_size=train_batch_size)
    val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # customized metrics
    #metric = load_metric("rouge")
    #rouge_metric = evaluate.load("./metric/rouge.py")
    rouge_metric = load_metric("./metric/rouge.py")
    metric_bert_score = BERTScorer(lang="en")

    def compute_rouge_metrics(generated_tokens, labels):
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def compute_hit_metrics(generated_tokens, labels):
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        hint_precisions, hint_hit_key_nums, hint_key_nums = [], [], []
        hint_recalls, hint_hit_word_nums, hint_word_nums = [], [], []
        assert len(labels) == len(decoded_preds)
        for i in range(len(decoded_preds)):
            pred = decoded_preds[i].strip().lower()
            pred = pred[:-1] if pred[-1] == "." else pred
            pred = pred.split(SPLIT.strip())
            pred = sorted(pred, key=lambda x: len(x), reverse=True)

            label = labels[i]
            label = label[label != 0]

            label_key = convert_label_to_key(label)
            label_summary = val_test_summary_mapping[label_key].lower()

            hit_pred = []
            for p in pred:
                p = p.strip()
                # " ".join(hit_pred)是将hit_pred中的所有单词用空格连接，形成的字符串
                if p not in " ".join(hit_pred) and p in label_summary and p not in avoid_keywords:
                    hit_pred.append(p)

            n = len(pred)
            hit_num = len(hit_pred)
            hit_precision = hit_num / n if n > 0 else 0
            print('预测关键词的数量', n)
            print('预测关键词中——命中关键词的数量', hit_num)
            print('命中关键词精确率hit_precision', hit_precision)


            hit_pred_word = []
            for p in " ".join(pred).split():
                p = p.strip()
                if p not in " ".join(hit_pred_word) and p in label_summary and p not in avoid_keywords:
                    hit_pred_word.append(p)

            m = len(label_summary.split())
            hit_word_num = len(hit_pred_word)
            hit_word_recall = hit_word_num / m if m > 0 else 0
            print('summary单词数量', m)
            print('目标摘要中——命中单词数量', hit_word_num)
            print('命中率单词召回率hit_recall', hit_word_recall)

            # store results
            hint_precisions.append(hit_precision)
            hint_hit_key_nums.append(hit_num)
            hint_key_nums.append(n)

            hint_recalls.append(hit_word_recall)
            hint_hit_word_nums.append(hit_word_num)
            hint_word_nums.append(m)


        result = {"hint_hit_key_num": np.mean(hint_hit_key_nums), "hint_key_precision": np.mean(hint_precisions), "key_num": np.mean(hint_key_nums),
                  "hint_hit_word_num": np.mean(hint_hit_word_nums), "hint_word_recall": np.mean(hint_recalls), "word_num": np.mean(hint_word_nums)}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in generated_tokens]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}


    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    #progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    running_loss = 0.0

    for epoch in range(starting_epoch, epochs):
        model.train()
        if with_tracking:
            total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                print('batch_loss:',loss)

                running_loss += loss.item()
                # for every 100 mini-batches, print the loss
                if step % 100 == 0:
                    print("epoch {} - step {}: average train loss {:.3f}".format(epoch, step, running_loss / 100))
                    running_loss = 0.0

                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                    print('total_loss:', total_loss)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_file = f"step_{completed_steps}"
                output_file = os.path.join(output_dir, output_file)
                accelerator.save_model(output_file)


        if completed_steps >= max_train_steps:
            break

        model.eval()
        set_seed(0)
        gen_kwargs = {
            "max_length": max_tgt_len,
            "min_length": 10,
        }

        all_predictions = []
        all_references = []
        all_predictions_token = []
        all_references_token = []
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs)
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)

                labels = batch["labels"]

                if not pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                if ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)


                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
                decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]


                all_predictions_token.extend(generated_tokens)
                all_references_token.extend(labels)
                all_predictions.extend(decoded_preds)
                all_references.extend(decoded_labels)


        if task == 'summarization':
            result_rouge= compute_rouge_metrics(all_predictions_token, all_references_token)
            logger.info(result_rouge)

            result_bert_score = {}
            precision, recall, f1 = metric_bert_score.score(cands=all_predictions, refs=all_references)
            result_bert_score["BERTScore Precision"] = round(precision.mean().item(), 4)
            result_bert_score["BERTScore Recall"] = round(recall.mean().item(), 4)
            result_bert_score["BERTScore F1"] = round(f1.mean().item(), 4)
            logger.info(result_bert_score)

        elif task == 'extraction':
            result_hit = compute_hit_metrics(all_predictions_token, all_references_token)
            # best_metric = "loss"
            logger.info(result_hit)

        if with_tracking:
            result_hit["train_mean_loss"] = total_loss.item() / len(train_dataloader)
            result_hit["epoch"] = epoch
            result_hit["step"] = completed_steps
            accelerator.log(result_hit, step=completed_steps)


        # 检查是否达到最大epoch
        if epoch == epochs - 1:
            if task == 'summarization':
                if isinstance(all_predictions, list):
                    predictions = [''.join(prediction.split('\n')) for prediction in all_predictions]
                    predictions_df = pd.DataFrame({'Prediction_summarization': predictions})
                    # 将结果保存到CSV文件
                    save_dir = f"./sft4lms/ckpt/{dataset}_{n_train}_{max_ctx_len}_{max_tgt_len}/{extraction_mode}-{extraction_source}/{model_name}-ep{epochs}"
                    csv_file_path = os.path.join(save_dir, f"test.csv")
                    predictions_df.to_csv(csv_file_path, index=False)
                    print(f"Predictions saved to '{csv_file_path}'")
                    # print(predictions_df)
                else:
                    raise ValueError("Predictions should be a list. Unable to save to CSV.")

            if task == 'extraction':
                if isinstance(all_predictions, list):
                    predictions = [''.join(prediction.split('\n')) for prediction in all_predictions]
                    predictions_df = pd.DataFrame({'Prediction_extraction': predictions})
                    # 将结果保存到CSV文件
                    # save_dir = f"./sft4lms/ckpt/{dataset}_{n_train}_{max_ctx_len}_{max_tgt_len}/{extraction_mode}-{extraction_source}/{model_name}-ep{epochs}"
                    # csv_file_path = os.path.join(save_dir, f"test.csv")
                    # predictions_df.to_csv(csv_file_path, index=False)
                    # print(f"Predictions saved to '{csv_file_path}'")
                    # print(predictions_df)
                else:
                    raise ValueError("Predictions should be a list. Unable to save to CSV.")

            if checkpointing_steps == "epoch":
                output_file = f"best_checkpoint"
                save_dir = os.path.join(output_dir, output_file)
                accelerator.save_state(save_dir)
                #accelerator.save_model(output_dir)

        else:
            print("Not the last epoch. No predictions or CSV file created.")


        if output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )



def main():

    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='cnndm', choices=['cnndm','general','attack']) #

    parser.add_argument('--n_train', type=int, default=2000) #
    parser.add_argument('--n_val', type=int, default=500) #
    parser.add_argument('--n_test', type=int, default=500) #
    parser.add_argument('--extraction_mode', type=str, default='textrank', choices=['textrank', 'patternrank', 'keybert', 'yake', 'prompt'])
    parser.add_argument('--extraction_source', type=str, default='all', choices=['all', 'article', 'summary'])
    #max_ctx_len输入长度，max_tgt_len输出长度，可设置
    parser.add_argument('--max_ctx_len', type=int, default=512)
    parser.add_argument('--max_tgt_len', type=int, default=64)

    # arguments for huggingface training
    parser.add_argument('--model', type=str, default='t5-base')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--load_strategy', type=str, default='load_initial', choices=['load_initial', 'load_best', 'load_last']) #

    parser.add_argument('--seed', type=int, default=1799)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--length_penalty', type=float, default=2)
    parser.add_argument('--no_repeat_ngram_size', type=float, default=3)
    # training task
    parser.add_argument('--task', type=str, default='extraction', choices=['extraction', 'summarization'])

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--push_to_hub', action='store_true')

    # 新增
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    parser.add_argument("--pad_to_max_length", action = "store_true")
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking",action="store_true")
    parser.add_argument("--report_to",type=str,default="all")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    

    args, unknown = parser.parse_known_args()

    # send_example_telemetry("run_extraction", args)
    model_path = args.model
    model_name = model_path.split('/')[-1]

    if args.output_dir is None:
        if args.task == 'summarization':
            output_dir = f"./sft4lms/ckpt/{args.dataset}_{args.n_test}_{args.max_ctx_len}_{args.max_tgt_len}/summarization/{model_name}/"
        elif args.task == 'extraction':
            output_dir = f"./sft4lms/ckpt/{args.dataset}_{args.n_test}_{args.max_ctx_len}_{args.max_tgt_len}/{args.extraction_mode}-{args.extraction_source}/{model_name}-ep{args.epochs}"
    else:
        output_dir = args.output_dir

    # LOAD cnndm DATA：/bin/zsh ./run_sft_cnndm.sh
    # Ptrain, Pval, Ptest = get_data_split(args.dataset, args.n_train, args.n_val, args.n_test, args.extraction_mode, args.extraction_source)
    # LOAD general lDATA：/bin/zsh ./run_sft_general.sh
    Ptrain, Pval, Ptest = get_general_data_split(args.dataset, args.n_train, args.n_val, args.n_test, args.extraction_mode, args.extraction_source)
    # LOAD attack lDATA：/bin/zsh ./run_sft_attack.sh
    # Ptrain, Pval, Ptest = get_attack_data_split(args.dataset, args.n_train, args.n_val, args.n_test, args.extraction_mode, args.extraction_source)

    # LOAD MODEL
    model, tokenizer = load_model(output_dir, model_path, args.load_strategy)

    assert args.max_ctx_len <= 1024 # for T5

    # if args.model in ['t5-base', 't5-large', 't5-3b']:
    #     model_path = args.model
    # elif args.model in ['flan-t5-large', 'flan-t5-base', 'flan-t5-small']:
    #     model_path = f"google/{args.model}"

    fine_tune_hf(
    task=args.task,
    model_name=model_name,
    dataset_name=args.dataset,
    n_train=args.n_train,
    push_to_hub=args.push_to_hub,
    model=model,
    tokenizer=tokenizer,
    extraction_source=args.extraction_source,
    max_ctx_len=args.max_ctx_len,
    max_tgt_len=args.max_tgt_len,
    output_dir=output_dir,
    train_data=Ptrain,
    val_data=Pval,
    test_data=Ptest,
    pad_to_max_length=args.pad_to_max_length,
    epochs=args.epochs,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    seed=args.seed,
    num_warmup_steps=args.num_warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_train_steps=args.max_train_steps,
    checkpointing_steps=args.checkpointing_steps,
    ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
    with_tracking=args.with_tracking,
    report_to=args.report_to,
    weight_decay=args.weight_decay
    )


if __name__ == "__main__":
    main()


import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse
import numpy as np
from sklearn.metrics import *
import nltk
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from datasets import load_dataset, load_metric, Dataset
from bert_score import BERTScorer
import evaluate

from sft4lms.Summarization.data_loader import get_data_split, get_general_data_split, get_attack_data_split, EXTRACTION_PREFIX, SUMMARIZATION_PREFIX, SPLIT
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords

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
    length_penalty,
    no_repeat_ngram_size,
    output_dir,
    train_data,
    val_data,
    test_data,
    epochs,
    train_batch_size,
    eval_batch_size,
    logging_steps,
    save_total_limit,
    early_stopping_patience,
    learning_rate,
    seed,
    do_train,
    do_inference
):  
    
    def preprocess_function_for_summarization(batch):
        inputs = [SUMMARIZATION_PREFIX + doc for doc in batch["article"]]
        targets = [doc for doc in batch["summary"]]
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_for_extraction(batch):
        #将batch中的 "article" 列中的每个文档前面加上 EXTRACTION_PREFIX = "Extract the keywords: "。
        inputs = [EXTRACTION_PREFIX + doc for doc in batch["article"]]
        targets = [doc for doc in batch["target"]]
        #输入文档进行分词，确保不超过最大长度 max_ctx_len
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        #目标文档进行分词，并确保不超过最大长度 max_tgt_len
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        #模型输入包括处理后的输入文档和目标
        # for input in inputs:
        #     print('len(input):',len(input))
        # for target in targets:
        #     print('len(target):',len(target))
        #     #print('target:', target)
        # for input in model_inputs['input_ids']:
        #     print('len(model_inputs):',len(input))
        # for target in model_inputs["labels"]:
        #     print('len(model_labels):',len(target))
            #print('model_labels:', target)
        #print("targets——训练", targets)

        return model_inputs

    #将输入的标签列表中的每个元素都转换为字符串，并使用空格连接起来，形成一个字符串键。这样，原始的标签列表中的每个元素都被空格分隔并组成了一个唯一的键。
    def convert_label_to_key(labels):
        return " ".join([str(i) for i in labels])
    
    def dataset_summary_mapping(dataset):
        mapping_dict = {}
        for d in dataset:
            #获取当前数据的标签，也就是源数据中的"target"，是一个字符串,比如：Mr Paxman; Mr Miliband; Mr Marr; Labour leader Ed Miliband; Mr Cameron.
            label = d['labels']
            #调用 convert_label_to_key 函数，将标签转换为特定的键（label_key）
            #如果输入的标签列表是 [1, 2, 3]，则通过该函数转换后的键为 "1 2 3"
            label_key = convert_label_to_key(label)
            #print('label_key', label_key)
            #将经过转换后的标签作为键，将数据的摘要（summary）作为值，添加到mapping_dict字典中。
            mapping_dict[label_key] = d['summary']
            print(mapping_dict)
        return mapping_dict

    # training the model with Huggingface 🤗 trainer
    #{"article": article, "summary": summary, "id": id, "phrases": selected_phrases, "target": target}
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)
    # 从训练数据集中移除名为 "phrases" 的列。phrases是关键词列表，这个在数据中的作用是什么？
    train_dataset = train_dataset.map(remove_columns=["phrases"])
    val_dataset = val_dataset.map(remove_columns=["phrases"])
    test_dataset = test_dataset.map(remove_columns=["phrases"])
    # tokenize the dataset
    if task == 'summarization':
        train_dataset = train_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["target", "article", "summary"])
        val_dataset = val_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["target", "article"])
        test_dataset = test_dataset.map(preprocess_function_for_summarization, batched=True, remove_columns=["target", "article"])
    elif task == 'extraction':
        # 应用预处理函数之前，将从数据集中移除名remove_columns列
        # 模型输入train_dataset又增加了分词器对处理后的输入文档[“article”]和目标[“target”]
        train_dataset = train_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["target", "article", "summary"])
        val_dataset = val_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["target", "article"])
        test_dataset = test_dataset.map(preprocess_function_for_extraction, batched=True, remove_columns=["target", "article"])

    # print('train_dataset:',train_dataset)
    # train_dataset: Dataset({
    #     features: ['id', 'input_ids', 'attention_mask', 'labels'],
    #     num_rows: 10
    # })
    # print('val_dataset:', val_dataset)
    # val_dataset: Dataset({
    #     features: ['summary', 'id', 'input_ids', 'attention_mask', 'labels'],
    #     num_rows: 5
    # })

    # mapping from label to summaries
    #数据集中的标签映射到对应的摘要，即通过标签查找相应的摘要信息。
    val_summary_mapping = dataset_summary_mapping(val_dataset)
    test_summary_mapping = dataset_summary_mapping(test_dataset)
    #两个字典（val_summary_mapping和test_summary_mapping）合并为一个新的字典val_test_summary_mapping。**符号用于将字典解包并合并到新的字典中。
    val_test_summary_mapping = {**val_summary_mapping, **test_summary_mapping}
    # remove unused columns summary，得到的 val_dataset、test_dataset 将是没有 "summary" 列的新数据集。
    val_dataset = val_dataset.map(remove_columns=["summary"])
    test_dataset = test_dataset.map(remove_columns=["summary"])


    #print('train_dataset', train_dataset)
    # train_dataset
    # Dataset({
    #     features: ['id', 'input_ids', 'attention_mask', 'labels'],
    #     num_rows: 10
    # })
    # print('val_dataset', val_dataset)
    # val_dataset
    # Dataset({
    #     features: ['id', 'input_ids', 'attention_mask', 'labels'],
    #     num_rows: 5
    # })
    # customized metrics
    #metric = load_metric("rouge")
    #rouge_metric = evaluate.load("./metric/rouge.py")
    rouge_metric = load_metric("./metric/rouge.py")

    all_predictions = []
    all_references = []
    def compute_rouge_metrics(eval_pred):

        # print('eval_pred [是id]：', eval_pred)
        predictions, labels = eval_pred
        #skip_special_tokens=True，将在解码过程中跳过特殊标记，比如起始标记（例如 [CLS] 或 <s>）和终止标记（例如 [SEP] 或 </s>）。
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print('decoded_preds0：', decoded_preds)
        print('decoded_labels0：', decoded_labels)
        
        # Rouge expects a newline after each sentence
        # strip() 方法用于移除字符串两侧的空白字符。
        # 将通过 sent_tokenize 分割得到的句子列表重新组合成一个字符串，每个句子之间用换行符 \n 连接。
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        print('decoded_preds1：', decoded_preds)
        print('decoded_labels1：', decoded_labels)
        
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}


        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        #添加 bert_score
        all_predictions.extend(decoded_preds)
        all_references.extend(decoded_labels)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    def compute_hit_metrics(eval_pred):

        # print('eval_pred [是id]：',eval_pred)
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        hint_precisions, hint_hit_key_nums, hint_key_nums = [], [], []
        hint_recalls, hint_hit_word_nums, hint_word_nums = [], [], []
        assert len(labels) == len(decoded_preds)
        for i  in range(len(decoded_preds)):
            #decoded_preds: 这是一个包含预测结果的列表或数组。每个元素应该是一个字符串，因为接下来的操作是对字符串进行处理。
            #strip()用于去除字符串两端的空格和换行符。
            pred = decoded_preds[i].strip().lower()
            #print('pred0：', pred)
            #pred[-1]: 这表示取字符串pred的最后一个字符。条件判断，检查最后一个字符是否是句号（"."）
            #如果 pred 的最后一个字符是句号，那么将 pred 中除了最后一个字符之外的所有字符赋给 pred。
            #如果 pred 的最后一个字符不是句号，那么 pred 的值保持不变。
            pred = pred[:-1] if pred[-1] == "." else pred
            # if pred and pred[-1] == ".":
            #     pred = pred[:-1]
            #使用SPLIT = "; " 作为分隔符，将字符串分割成一个列表（数组）
            pred = pred.split(SPLIT.strip())
            #print('pred1：', pred)
            # remove the repeated words,把 pred 中的字符串按照长度从长到短排序。
            pred = sorted(pred, key=lambda x: len(x), reverse=True)
            print('pred2：', pred)
            # label -> summary
            label = labels[i]

            # 表示从 label 中移除所有值为 -100 的元素。在flan-t5-large -100 通常用于表示填充或无效的标签。
            label = label[label != -100]
            # 表示从 label 中移除所有值为 1 的元素。在bart-large-cnn 1 通常用于表示填充标签。
            # label = label[label != -100]
            # label = label[label!=1]
            #它将处理后的标签 label 转换为一个键，可能是用于查找某个映射或字典的关键字。
            label_key = convert_label_to_key(label)
            #取出与 label_key 对应的值,也就是summary
            label_summary = val_test_summary_mapping[label_key].lower()
            print('label_summary：', label_summary)#str

            #label = val_test_summary_mapping[label_key].lower()


            # hit_pred 列表将包含那些预测结果中命中标签 label 且不在避免关键词列表中的元素。
            # 这个过程可能是为了筛选出模型预测的结果中与真实标签匹配的关键词，同时排除一些不希望出现的关键词。
            hit_pred = []
            for p in pred:
                p = p.strip()
                # " ".join(hit_pred)是将hit_pred中的所有单词用空格连接，形成的字符串
                if p not in " ".join(hit_pred) and p in label_summary and p not in avoid_keywords:
                    hit_pred.append(p)

            # calculate hit score and precision
            # pred：预测的关键词数（关键词可能是一个单词、也可能是一组单词）；hit_num：命中的关键词数；hit_precision：命中精确率
            n = len(pred)
            hit_num = len(hit_pred)
            hit_precision = hit_num / n if n > 0 else 0
            print('预测关键词长度',n)
            print('命中关键词长度', hit_num)
            print('命中关键词精确率hit_precision', hit_precision)

            # # label：label的单词长度(单词数量)；hit_word_num：命中的单词数；hit_word_recall：命中召回率
            # m = len(label_summary.split())
            # hit_word_pred = " ".join(hit_pred).split()
            # hit_word_num = len(hit_word_pred)
            # hit_word_recall = hit_word_num / m if m > 0 else 0
            # print('label_summary单词长度', m)
            # print('命中单词长度', hit_word_num)
            # print('命中率单词召回率hit_recall', hit_word_recall)

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

        # Extract a few results
        #result = {"hint_hit_key_num": np.mean(hint_hit_key_nums), "hint_key_precision": np.mean(hint_precisions), "key_num": np.mean(hint_key_nums)}

        result = {"hint_hit_key_num": np.mean(hint_hit_key_nums), "hint_key_precision": np.mean(hint_precisions), "key_num": np.mean(hint_key_nums),
                  "hint_hit_word_num": np.mean(hint_hit_word_nums), "hint_word_recall": np.mean(hint_recalls), "word_num": np.mean(hint_word_nums)}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    # tokenize the dataset
    if task == 'summarization':
        compute_metrics = compute_rouge_metrics
        best_metric = "rouge1"

    elif task == 'extraction':
        compute_metrics = compute_hit_metrics
        best_metric = "loss"

    # arguments
    training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir, # output directory
    # output_dir=output_dir if not push_to_hub else hf_path, # output directory
    num_train_epochs=epochs, # total number of training epochs
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
    evaluation_strategy='steps',
    # length_penalty = length_penalty,
    # no_repeat_ngram_size = no_repeat_ngram_size,
    learning_rate=learning_rate, # 2e-5
    weight_decay=0.01,
    # fp16=True,
    # 设置这三个参数为相同的值可以简化训练配置，使得日志记录、评估和模型保存都在相同的频率下进行，更容易管理和理解整个训练过程。
    # 多少个训练步之后记录日志/评估模型/权重模型
    logging_steps=logging_steps, # the same as eval_step，如果 logging_steps 被设置为 100，那么在每训练 100 个步骤后，模型就会输出一次训练日志。
    eval_steps=logging_steps, # 在多少个训练步之后进行一次模型评估
    save_steps=logging_steps, # doesn't work if load_best_model_at_end=True, will save every eval_steps (logging_steps)
    logging_dir=os.path.join(output_dir, "runs/"),
    save_total_limit=save_total_limit,# save_total_limit=3，每当生成一个新的模型检查点时，系统会保留最新的 3 个检查点，而删除更旧的检查点。
    seed=seed,
    push_to_hub=push_to_hub,
    predict_with_generate=True, # for evaluation metrics
    # load_best_model_at_end=True,
    # metric_for_best_model=best_metric,
    remove_unused_columns=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)

    trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]    
    )
    
    if do_train:
        train_results = trainer.train()
        print('train_results',train_results)
        eval_results = trainer.evaluate()
        print('train_results',eval_results)
        # save model locally and push it to the hub
        trainer.save_model(os.path.join(output_dir, "best_checkpoint/"))
        print(f'Save best model in {os.path.join(output_dir, "best_checkpoint/")}')
        if push_to_hub:
            trainer.push_to_hub()

    # inference on the test set
    if do_inference:
        #test_results = trainer.predict(test_dataset)
        test_results = trainer.predict(test_dataset, min_length=5)
        print('test_results',test_results)
    
    return model


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
    # 输出目录在代码中定义
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
    
    args, unknown = parser.parse_known_args()

    dataset = args.dataset
    task = args.task
    load_strategy = args.load_strategy
    n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
    extraction_mode = args.extraction_mode
    extraction_source = args.extraction_source
    max_ctx_len = args.max_ctx_len
    max_tgt_len = args.max_tgt_len
    length_penalty = args.length_penalty
    no_repeat_ngram_size = args.no_repeat_ngram_size
    assert max_ctx_len <= 1024 # for T5
    # 在源代码代码中args.model是模型的名字，model_path是模型的下载路径有关
    # 在我们的代码里model_name是模型的名字，model_path是模型的路径
    model_path = args.model
    model_name = model_path.split('/')[-1]

    # if args.model in ['t5-base', 't5-large', 't5-3b']:
    #     model_path = args.model
    # elif args.model in ['flan-t5-large', 'flan-t5-base', 'flan-t5-small']:
    #     model_path = f"google/{args.model}"

    """prepare for training"""
    if args.output_dir is None:
        if task == 'summarization':
            output_dir = f"./sft4lms/ckpt/{dataset}_{n_train}_{max_ctx_len}_{max_tgt_len}/summarization/{model_name}/"
        elif task == 'extraction':
            output_dir = f"./sft4lms/ckpt/{dataset}_{n_train}_{max_ctx_len}_{max_tgt_len}/{extraction_mode}-{extraction_source}/{model_name}-ep{args.epochs}"
    else:
        output_dir = args.output_dir

    # LOAD cnndm DATA：/bin/zsh ./run_sft_cnndm.sh
    # Ptrain, Pval, Ptest = get_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source)
    # LOAD general lDATA：/bin/zsh ./run_sft_general.sh
    Ptrain, Pval, Ptest = get_general_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source)
    # LOAD attack lDATA：/bin/zsh ./run_sft_attack.sh
    # Ptrain, Pval, Ptest = get_attack_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source)

    # LOAD MODEL
    model, tokenizer = load_model(output_dir, model_path, load_strategy)

    fine_tune_hf(
    task=args.task,
    # model_name=args.model,
    model_name=model_name,
    dataset_name=args.dataset,
    n_train=n_train,
    push_to_hub=args.push_to_hub,
    model=model,
    tokenizer=tokenizer,
    extraction_source=extraction_source,
    max_ctx_len=max_ctx_len,
    max_tgt_len=max_tgt_len,
    length_penalty=length_penalty,
    no_repeat_ngram_size=no_repeat_ngram_size,
    output_dir=output_dir,
    train_data=Ptrain,
    val_data=Pval,
    test_data=Ptest,
    epochs=args.epochs,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    logging_steps=args.logging_steps,
    save_total_limit=args.save_total_limit,
    early_stopping_patience=args.early_stopping_patience,
    learning_rate=args.learning_rate,
    seed=args.seed,
    do_train=args.do_train,
    do_inference=args.do_inference
    )


if __name__ == "__main__":
    main()


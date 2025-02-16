import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json

from tqdm import trange
from tqdm import tqdm

import spacy
import datasets
from keybert import KeyBERT
import yake
import pytextrank
from transformers import AutoTokenizer

# from sft4lms.Summarization.utils import *
from rl4lms.envs.text_generation.gpt3_utils import GPT3, avoid_keywords

EXTRACTION_PREFIX = "Extract the keywords: "
SUMMARIZATION_PREFIX = "Summarize: "

#SPLIT是一个包含分号和空格的字符串，用于将已选择的短语连接起来时使用。
SPLIT = "; "

def dictlist2df(dict_list):
    dl = {}
    for d in dict_list:
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    df = pd.DataFrame.from_dict(dl)
    return df


def dictlist2dict(dict_list):
    dl = {}
    # 遍历dict_list中的每个元素，比如{"article": "article1", "summary": "summary1", "id": 1, "phrases": 'selected_phrases1', "target": "target1"}
    for d in dict_list:
        #遍历当前字典 d 中的键值对, 比如{"article": "article1"}
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    # dl是一个字典，字典中的每个元素是一个二维数组，'id': ['53a67', '30c60', 'd6654c', '23778', '76971'],
    return dl


def filter_keywords(phrases, summary, article):

    # sort according to lengths依据短语的长度进行排序
    phrases = sorted(phrases, key=lambda x: len(x))[::-1]

    phrase_indices = []
    selected_phrases = []
    #避开的单词，这个和我们的有相似之处
    avoid_phrases = ["one", "two", "three", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "an", "the", "he", "she", "i", "we", "you", "they", "it", "this",
                     "that", "those", "these", "me", "them", "him", "his", "her", "my", "your", "its", "ours", "our", "their", "what", "which", "who", "why", "when",
                     "whom", "whose", "could", "with", " ", ",", ".", "?", "!", ";","-","_","="]

    for p in phrases:
        
        if p.lower() in avoid_phrases: ## avoid phrases
            continue
        if p.lower() in SPLIT.join(selected_phrases).lower(): ## has already selected
            continue
        if p.lower() not in summary.lower(): ## not in summary
            continue
        if p.lower() not in article.lower(): ## not in article
            continue

        # place in the order of summary
        # 先将摘要和当前短语 p 转换为小写（.lower()），然后用index方法找到p在摘要中的第一个出现位置的索引。这个索引值被赋给变量 p_index。
        p_index = summary.lower().index(p.lower())

        selected_phrases.append(p)
        phrase_indices.append(p_index)

    # sort
    #将 selected_phrases 和 phrase_indices 两个列表按元素进行配对，形成一个由元组组成的新列表
    selected_phrases = list(zip(selected_phrases, phrase_indices))
    #对配对后的列表进行排序，排序的依据是元组中的第二个元素，即短语在摘要中的索引位置
    selected_phrases = sorted(selected_phrases, key=lambda x: x[1])
    #将排序后的元组列表重新转换为只包含短语的列表，保留了它们在摘要中的排序顺序
    selected_phrases = [p[0] for p in selected_phrases]

    #selected_phrases 列表包含了这些短语，并按照它们在摘要中的出现顺序进行了重新排列。
    return selected_phrases


def get_extraction_data(dataset, data, extraction_mode, extraction_source):
    # eg. get_extraction_data(cnndm, train_data, textrank, all)

    # Init
    if extraction_mode == 'textrank':
        # 使用 spaCy 库加载了英语（"en_core_web_sm"）的自然语言处理模型
        # textrank包含了分词、命名实体识别、词性标注等自然语言处理功能
        # nlp 对象是 spaCy 中的核心对象，它可以用于处理文本数据。
        nlp = spacy.load("en_core_web_sm")
        # 向 spaCy 的处理流水线（pipeline）中添加了 TextRank 算法。TextRank 是一个用于关键词抽取和文本摘要的图算法。
        # 通过将 TextRank 添加到处理流水线中，后续使用 nlp 对象时，让文本经过 TextRank 处理，得到关键词或摘要信息。
        nlp.add_pipe("textrank")
    elif extraction_mode == 'yake':
        #YAKE!基于5种指标：是否大写，词的位置，词频，上下文关系，词在句中频率，来计算候选词的得分，从而筛选Top-N关键词。
        kw_extractor = yake.KeywordExtractor()
    elif extraction_mode == 'prompt':
        #gpt3 = GPT3(2.0) # sleep 5s before each call
        gpt3 = GPT3(model="gpt-3.5-turbo")  # sleep 5s before each call
        keyword_prompt_path = f"./prompts/{dataset}_keyword_fs.txt"
        f = open(keyword_prompt_path, 'r') 
        keyword_prompt = f.read().strip()
        stop_words = ["\n", "Sentences:"]

    # Start
    processed_data = []
    # 记录每个文章关键词的个数
    phrase_num = []
    for d in trange(len(data)):

        article = data[d]['article']
        summary = data[d]['highlights']
        id = data[d]['id']
        selected_phrases = []

        # print('article:',article)
        # print('summary:', summary)
        # print('id:', id)


        #如果其中任何一个为空（即为假），则继续下一次循环，跳过当前文章的处理。这是为了确保文章和摘要都不为空，以避免处理不完整的数据。
        if not article or not summary:
            continue

        # extract from summary or article
        if extraction_source == "article":
            source = article
        elif extraction_source == "summary":
            source = summary
        elif extraction_source == "all":
            source = article + "\n" + summary
        else:
            raise NotImplementedError


        """
        Step 1: extraction
        """
        # Extraction with textrank
        if extraction_mode == 'textrank':
            # doc包含了各种语言处理信息，如分词、词性标注、命名实体识别等
            doc = nlp(source)
            #selected_phrases 存储关键短语的文本
            for phrase in doc._.phrases:
                selected_phrases.append(phrase.text)
            # selected_phrases一维单词列表
            #print('selected_phrases0:', d, selected_phrases, len(selected_phrases))

        # Extraction with yake
        elif extraction_mode == 'yake':
            phrases = kw_extractor.extract_keywords(source)
            for phrase in phrases:
                selected_phrases.append(phrase[0])
            #print('selected_phrases0:', d, selected_phrases, len(selected_phrases))
    
        # Extraction with GPT3 Prompt
        elif extraction_mode == 'prompt':
            input = keyword_prompt.replace("[[QUESTION]]", source)
            candidates = gpt3.call(messages=input,
                                temperature=0.7,
                                max_tokens=64,
                                n=1,
                                top_p=1.0,
                                stop=stop_words
                                )
            for candidate in candidates:
                for keyword in candidate[:-1].split(SPLIT):
                    selected_phrases.append(keyword.strip())
            selected_phrases = list(set(selected_phrases))
            #print('selected_phrases0:', d, selected_phrases, len(selected_phrases))

        # Not Implemented Yet...
        else:
            raise NotImplementedError()

        """
        Step 2: selection and tokenization
        """
        # sort phrases according to appearances in the article
        # selected_phrases 列表包含了summary, article出现的单词、短语，并按照它们在摘要中的出现顺序进行了重新排列。
        selected_phrases = filter_keywords(selected_phrases, summary, article)
        #print('selected_phrases1:', d, selected_phrases,len(selected_phrases))
        # 首先使用定义的分隔符 SPLIT="; " 将 selected_phrases 列表中的短语连接起来
        # 然后在连接后的字符串末尾添加一个句点。这样形成的字符串被赋给变量 target。
        # 如果 selected_phrases 中没有任何短语（即长度为0），则 target 被赋值为空字符串。
        target = SPLIT.join(selected_phrases) + "." if len(selected_phrases) > 0 else ""
        # print('selected_phrases:', selected_phrases)
        # print('target:',target)

        phrase_num.append(len(selected_phrases))

        # save data
        #selected_phrases是一个列表、target是一个字符串
        #selected_phrases是过滤后的关键短语列表，target 是通过连接这些关键短语形成的字符串，用于表示文章中的关键信息。
        processed_data.append({"article": article, "summary": summary, "id": id, "phrases": selected_phrases, "target": target})

    # Statistics
    len_min = np.min(phrase_num)
    len_mean = np.mean(phrase_num)
    len_median = np.median(phrase_num)
    len_max = np.max(phrase_num)
    print("mean of phrase num:{}, median of phrase num:{}, max of phrase num:{}, min of phrase num {}".format(len_mean, len_median, len_max, len_min))

    #print("target", target)
    return processed_data


def get_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source, keyword_num_range=(1, 20), return_dict=True):

    # load existing data
    load_path = f"./sft4lms/data/cnndm/sample_cnndm/"

    # save propessing data
    save_path = f"./sft4lms/data/{dataset}/{extraction_mode}-{extraction_source}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # training data
    train_raw_data = pd.read_parquet(load_path + "sample_train.parquet")
    train_selected_data = []
    for idy, d_ in train_raw_data.iterrows():
        train_selected_data.append(d_)
    train_data = get_extraction_data(dataset, train_selected_data, extraction_mode, extraction_source)
    random.shuffle(train_data)
    np.save(save_path + "train.npy", train_data)

    # test data, select 500 subset
    test_raw_data = pd.read_parquet(load_path + "sample_test.parquet")
    # test_selected_data的类型为list
    test_selected_data = []

    with open(f"./sft4lms/data/{dataset}/{dataset}_test500.json", 'r') as file:
        test_data_subset = json.load(file)
    for idx, d in test_data_subset.items():
        for idy, d_ in test_raw_data.iterrows():
            if d_['id'] == d['id']:
                test_selected_data.append(d_)
                break
    # get extraction data
    test_data = get_extraction_data(dataset, test_selected_data, extraction_mode, extraction_source)
    np.save(save_path + "test.npy", test_data)

    # validation data
    val_raw_data = pd.read_parquet(load_path + "sample_val.parquet")
    val_selected_data = []
    with open(f"./sft4lms/data/{dataset}/{dataset}_val500.json", 'r') as file:
        val_data_subset = json.load(file)
    for idx, d in val_data_subset.items():
        for idy, d_ in val_raw_data.iterrows():
            if d_['id'] == d['id']:
                val_selected_data.append(d_)
                break
    # get extraction data
    val_data = get_extraction_data(dataset, val_selected_data, extraction_mode, extraction_source)
    np.save(save_path + "val.npy", val_data)
    # print(len(val_selected_data))
    # print('get_extraction_data_val_data', val_data)

    # select n samples，确保train_data 的长度为n_train 和 train_data 长度中的较小值
    train_data = train_data[:min(n_train, len(train_data))]
    val_data = val_data[:min(n_val, len(val_data))]
    test_data = test_data[:min(n_test, len(test_data))]


    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(val_data), dictlist2dict(test_data)
    else:
        return train_data, val_data, test_data

def get_general_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source, return_dict=True):

    # load existing data
    # load_path = f"./sft4lms/data/{dataset}/sample_general/"
    load_path = f"./sft4lms/data/{dataset}/general_parquet/"

    # save propessing data
    save_path = f"./sft4lms/data/{dataset}/{extraction_mode}-{extraction_source}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # training data
    train_raw_data = pd.read_parquet(load_path + "train.parquet")
    train_selected_data = []
    for idy, d_ in train_raw_data.iterrows():
        train_selected_data.append(d_)
    train_data = get_extraction_data(dataset, train_selected_data, extraction_mode, extraction_source)
    random.shuffle(train_data)
    np.save(save_path + "train.npy", train_data)

    # testing data
    test_raw_data = pd.read_parquet(load_path + "test.parquet")
    test_selected_data = []
    for idy, d_ in test_raw_data.iterrows():
        test_selected_data.append(d_)
    test_data = get_extraction_data(dataset, test_selected_data, extraction_mode, extraction_source)
    random.shuffle(test_data)
    np.save(save_path + "test.npy", test_data)

    # training data
    valid_raw_data = pd.read_parquet(load_path + "valid.parquet")
    valid_selected_data = []
    for idy, d_ in valid_raw_data.iterrows():
        valid_selected_data.append(d_)
    valid_data = get_extraction_data(dataset, valid_selected_data, extraction_mode, extraction_source)
    random.shuffle(valid_data)
    np.save(save_path + "valid.npy", valid_data)

    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(valid_data), dictlist2dict(test_data)
    else:
        return train_data, valid_data, test_data

def get_attack_data_split(dataset, n_train, n_val, n_test, extraction_mode, extraction_source, return_dict=True):

    # load existing data
    # load_path = f"./sft4lms/data/{dataset}/sample_parquet/"
    load_path = f"./sft4lms/data/{dataset}/attack_parquet/"
    # save propessing data
    save_path = f"./sft4lms/data/{dataset}/{extraction_mode}-{extraction_source}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # training data
    train_raw_data = pd.read_parquet(load_path + "train.parquet")
    train_selected_data = []
    for idy, d_ in train_raw_data.iterrows():
        train_selected_data.append(d_)
    train_data = get_extraction_data(dataset, train_selected_data, extraction_mode, extraction_source)
    random.shuffle(train_data)
    np.save(save_path + "train.npy", train_data)

    # testing data
    test_raw_data = pd.read_parquet(load_path + "test.parquet")
    test_selected_data = []
    for idy, d_ in test_raw_data.iterrows():
        test_selected_data.append(d_)
    test_data = get_extraction_data(dataset, test_selected_data, extraction_mode, extraction_source)
    random.shuffle(test_data)
    np.save(save_path + "test.npy", test_data)

    # training data
    valid_raw_data = pd.read_parquet(load_path + "valid.parquet")
    valid_selected_data = []
    for idy, d_ in valid_raw_data.iterrows():
        valid_selected_data.append(d_)
    valid_data = get_extraction_data(dataset, valid_selected_data, extraction_mode, extraction_source)
    random.shuffle(valid_data)
    np.save(save_path + "valid.npy", valid_data)

    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(valid_data), dictlist2dict(test_data)
    else:
        return train_data, valid_data, test_data

if __name__ == "__main__":

    for dataset in ['cnndm']:
        for extraction_mode in ['textrank']:
            for extraction_source in ['all']:
                for n_train in [1000, 2000, 4000]:
                    for n_val in [500]:
                        for n_test in [500]:
                            train_data, val_data, test_data = get_data_split(dataset=dataset, n_train=n_train, n_val=n_val, n_test=n_test,
                                                                            extraction_mode=extraction_mode, extraction_source=extraction_source, 
                                                                            keyword_num_range=(1, 20))

                            from datasets import Dataset
                            train_dataset = Dataset.from_dict(train_data)
                            val_dataset = Dataset.from_dict(val_data)
                            test_dataset = Dataset.from_dict(test_data)
                            tokenizer = AutoTokenizer.from_pretrained("gpt2")

                            phrase_nums = []
                            article_lens = []
                            summary_lens = []
                            target_lens = []
                            analyzed_dataset = train_dataset
                            
                            for i in range(len(analyzed_dataset)):
                            # for i in random.sample(range(len(dataset)), 100):
                                phrase_nums.append(len(analyzed_dataset[i]['phrases']))
                                article = analyzed_dataset[i]['article']
                                summary = analyzed_dataset[i]['summary']
                                target = analyzed_dataset[i]['target']
                                article_lens.append(len(tokenizer(article).input_ids))
                                summary_lens.append(len(tokenizer(summary).input_ids))
                                target_lens.append(len(tokenizer(target).input_ids))

                                # # # DEBUG
                                # print("Index:")
                                # print(i)

                                # print("Keywords:")
                                # print(analyzed_dataset[i]['phrases'])

                                # print("Target:")
                                # print(analyzed_dataset[i]['target'])

                                # _ = input("continue.........")



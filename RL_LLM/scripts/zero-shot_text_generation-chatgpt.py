from rl4lms.envs.text_generation.gpt3_utils import GPT3
from transformers import GPT2TokenizerFast
from typing import List
import pandas as pd
import numpy as np
import json


#根据给定的停止词列表，截取文本字符串中出现在停止词之前的部分，并返回处理后的结果。
def clean_generation(text, stop_words):
        text = text.strip()
        end_idx = len(text)
        for end_word in stop_words:
            if end_word in text:
                end_idx = min(end_idx, text.find(end_word))
        text = text[:end_idx]
        return text
    
def gpt3_hint_generation(gpt3,
                             input: str, 
                             temperature: float,
                             max_tokens: int,
                             num_seqs: int,
                             top_p: float,
                             stop_words: List[str]):
            candidates = gpt3.call(prompt=input,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    n=num_seqs,
                                    top_p=top_p,
                                    stop=stop_words
                                    )

            candidates = [clean_generation(candidate, stop_words+["\n\n", "\n"]) for candidate in candidates]
            return candidates

        
def generation_selection(strategy: str, candidates: List[str]) -> List[str]:
        if strategy == 'lcs':
            from string2string.edit_distance import EditDistAlgs
            algs_unit = EditDistAlgs()
            n = len(candidates)
            matrix = np.zeros((n,n))
            for j1, cand1 in enumerate(candidates):
                cand1_split = cand1.split(' ')
                for j2, cand2 in enumerate(candidates):
                    cand2_split = cand2.split(' ')
                    max_length = max(len(cand1_split), len(cand2_split))
                    dist, _ = algs_unit.longest_common_subsequence(
                        cand1_split,
                        cand2_split,
                        printBacktrack=False,
                        boolListOfList=True
                        )
                    score = dist / max_length
                    matrix[j1][j2] = score
            matrix = np.mean(matrix, axis=1)
            index = np.argmax(matrix)
            return [candidates[index]]
        
        elif strategy == 'choose_first':
            return [candidates[0]]
        
        elif strategy == 'choose_all':
            return candidates
        
        elif strategy == 'random':
            index = np.random.randint(0, len(candidates))
            return [candidates[index]]
        
        return candidates

    
gpt3_model = 'gpt-3.5-turbo'
interval = 0.5
timeout = 35.0
exp = 2.0
patience = 10
temperature = 0.7
max_tokens = 512
num_seqs = 1
selection_strategy = "choose_all"
top_p = 1.0
stop_words = ["Article:", "Q:",  "A:", "<|im_end|>"]

#prompt_path = "./prompts/general_fs.txt"
prompt_path = "/root/autodl-tmp/RL_LLM/prompts/attack_fs.txt"
hint_prompt_path = "/root/autodl-tmp/RL_LLM/prompts/attack_hint_fs.txt"
split_token = ";"
split_token_id = 3

f = open(prompt_path, 'r') 
prompt = f.read().strip()
f = open(hint_prompt_path, 'r') 
hint_prompt = f.read().strip()

gpt3 = GPT3(model=gpt3_model, interval=interval, timeout=timeout, exp=exp, patience=patience)
tokenizer = GPT2TokenizerFast.from_pretrained("/root/autodl-tmp/RL_LLM/pretrain_model/gpt2")
   

# 读取npy
# data = np.load(r'/root/autodl-tmp/RL_LLM/sft4lms/data/general/textrank-all_flan_t5-base_pred/step1_key_article.npy', allow_pickle = True)
# # 转为list
# data = data.tolist()


# # 写入 JSON 文件   输入==article
# with open('/root/autodl-tmp/RL_LLM/sft4lms/data/general/textrank-all_flan_t5-base_pred/step1_key_article.json', 'w', encoding = "utf-8") as f:
#     num =0
#     for item in data:
#         article = item['article']
        
#         article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
#         article = tokenizer.decode(article_ids)

#         gpt3_input_text = prompt.replace("[[QUESTION]]", article)
#         gpt3_gen_texts = gpt3_hint_generation(gpt3, gpt3_input_text, temperature, max_tokens, num_seqs, top_p, stop_words)
#         gpt3_gen_texts_article = generation_selection(selection_strategy, gpt3_gen_texts)

#         print('完成～', gpt3_gen_texts_article)
#         item['gpt3_gen_texts'] = gpt3_gen_texts_article
#         print('完成～'+ str(num))
#         num += 1
#     json.dump(data, f, indent=4)
#     f.close()
# print('完成～')


# 读取csv 输入==article
# data_frame = pd.read_csv('/root/autodl-tmp/RL_LLM/sft4lms/data/attack/textrank-all_flan_t5-base_pred/valid.csv')
    
# # 对每一行进行操作并保存结果
# result_list = []
# num =0
# for index, row in data_frame.iterrows():
#     article = row['article']
    
#     article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
#     article = tokenizer.decode(article_ids)

#     gpt3_input_text = prompt.replace("[[QUESTION]]", article)
    
#     gpt3_gen_texts = gpt3_hint_generation(gpt3, gpt3_input_text, temperature, max_tokens, num_seqs, top_p, stop_words)
#     gpt3_gen_texts_article = generation_selection(selection_strategy, gpt3_gen_texts)
    
#     row['gpt3_gen_texts'] = gpt3_gen_texts_article
#     print('完成～', gpt3_gen_texts_article)
#     print('完成～'+ str(num))
#     num += 1
#     result_dict = {'article': row['article'], 'summary': row['summary'], 'gpt3_gen_texts' :row['gpt3_gen_texts']}
#     result_list.append(result_dict)

# # 将结果列表转换为 JSON 格式并写入文件
# json_file_path = "/root/autodl-tmp/RL_LLM/sft4lms/data/attack/textrank-all_flan_t5-base_pred/valid_article.json"
# with open(json_file_path, 'w') as json_file:
#     json.dump(result_list, json_file, indent=4) 



    
#读取csv 输入==article+target
# data_frame = pd.read_csv('/root/autodl-tmp/RL_LLM/sft4lms/data/attack/textrank-all_flan_t5-base_pred/valid.csv')
    
# # 对每一行进行操作并保存结果
# result_list = []
# num =0
# for index, row in data_frame.iterrows():
#     article = row['article']
#     article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
#     article = tokenizer.decode(article_ids)
    
#     target = row['prediction']
    
#     gpt3_input_text = hint_prompt.replace("[[QUESTION]]", article)
#     gpt3_input_text = gpt3_input_text.replace("[[HINT]]", target)
    
#     gpt3_hint_gen_texts = gpt3_hint_generation(gpt3, gpt3_input_text, temperature, max_tokens, num_seqs, top_p, stop_words)
#     gpt3_hint_gen_article_target = generation_selection(selection_strategy, gpt3_hint_gen_texts)

    
#     row['gpt3_gen_texts'] = gpt3_hint_gen_article_target
#     print('完成～', gpt3_hint_gen_article_target)
#     print('完成～'+ str(num))
#     num += 1
#     result_dict = {'article': row['article'], 'prediction': row['prediction'], 'summary': row['summary'], 'gpt3_gen_texts' :row['gpt3_gen_texts']}
#     result_list.append(result_dict)

# # 将结果列表转换为 JSON 格式并写入文件
# json_file_path = "/root/autodl-tmp/RL_LLM/sft4lms/data/attack/textrank-all_flan_t5-base_pred/valid_keyword_article.json"
# with open(json_file_path, 'w') as json_file:
#     json.dump(result_list, json_file, indent=4) 
    
    


# # 写入 JSON 文件   输入==generated_text+article   
with open('/root/autodl-tmp/RL_LLM/sft4lms/data/attack/textrank-all_flan_t5-base_pred/R_test_attack_predictions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('/root/autodl-tmp/RL_LLM/sft4lms/data/attack/textrank-all_flan_t5-base_pred/R_test_attack_predictions.json', 'w', encoding = "utf-8") as f:
    num =0
    for item in data:

        article = item['prompt_text']
        article = article.replace("Extract the keywords: ", "") 
        target = item['generated_text']
       
        article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
        article = tokenizer.decode(article_ids)
        
        # gpt3_input_text = hint_prompt.replace("[[HINT]]", target)
        # gpt3_input_text = gpt3_input_text.replace("[[QUESTION]]", article)
        
        gpt3_input_text = hint_prompt.replace("[[QUESTION]]", article)
        gpt3_input_text = gpt3_input_text.replace("[[HINT]]", target)
        
        
        gpt3_hint_gen_texts = gpt3_hint_generation(gpt3, gpt3_input_text, temperature, max_tokens, num_seqs, top_p, stop_words)
        gpt3_hint_gen_article_target = generation_selection(selection_strategy, gpt3_hint_gen_texts)
        
        item['gpt3_gen_texts2'] = gpt3_hint_gen_article_target
        print('完成～'+ str(num))
        num += 1
    
    json.dump(data, f, indent=4)
    f.close()
print('完成～')



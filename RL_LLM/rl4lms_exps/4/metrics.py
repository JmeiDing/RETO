from rouge import Rouge
from bert_score import score
import json
import nltk
from rouge_score import rouge_scorer
import numpy as np
import re

avoid_phrases = ["one", "two", "three", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "an", "the", "he", "she", "i", "we", "you", "they",
                 "it", "this", "that", "those", "these", "me", "them", "him", "his", "her", "my", "your", "its", "ours", "our", "their", "what",
                 "which", "who", "why", "when", "whom", "whose", "could", "with", " ", ",", ".", "?", "!", ";","-","_","="]
from rouge_score import rouge_scorer
import evaluate

def calculate_rouge(reference_texts_list, generated_texts_list):
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
    num_samples = len(reference_texts_list)

    for reference_texts, generated_texts in zip(reference_texts_list, generated_texts_list):
        scores = rouge_scorer_instance.score(generated_texts, reference_texts)
        #print("ROUGE scores:", scores)

        for metric in scores:
            rouge_scores_sum[metric] += scores[metric].fmeasure

        # Calculate ROUGE-Lsum
        rougeLsum = (2 * scores['rougeL'].precision * scores['rougeL'].recall) / (scores['rougeL'].precision + scores['rougeL'].recall + 1e-12)
        rouge_scores_sum['rougeLsum'] += rougeLsum


    # Calculate average ROUGE scores
    for metric in rouge_scores_sum:
        rouge_scores_sum[metric] /= num_samples


    return rouge_scores_sum

def calculate_rouge2(reference_texts_list, generated_texts_list):
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
    num_valid_samples = 0  # 用于记录有效样本数量

    for reference_texts, generated_texts in zip(reference_texts_list, generated_texts_list):
        # 检查生成的文本和参考文本是否为 None
        if reference_texts is None or generated_texts is None:
            continue  # 如果任一文本为 None，则跳过当前样本

        scores = rouge_scorer_instance.score(generated_texts, reference_texts)
        for metric in scores:
            rouge_scores_sum[metric] += scores[metric].fmeasure

        # Calculate ROUGE-Lsum
        rougeLsum = (2 * scores['rougeL'].precision * scores['rougeL'].recall) / (scores['rougeL'].precision + scores['rougeL'].recall + 1e-12)
        rouge_scores_sum['rougeLsum'] += rougeLsum

        num_valid_samples += 1

    # Calculate average ROUGE scores
    if num_valid_samples > 0:
        for metric in rouge_scores_sum:
            rouge_scores_sum[metric] /= num_valid_samples
    else:
        print("Warning: No valid samples for ROUGE calculation.")

    return rouge_scores_sum


def calculate_bert_score(reference_texts, candidate_texts):
    P, R, F1 = score(reference_texts, candidate_texts, lang='en', verbose=False)
    bert_scores = {'precision': round(P.mean().item(), 5), 'recall': round(R.mean().item(), 5), 'f1': round(F1.mean().item(), 5)}
    return bert_scores

def calculate_bert_score2(reference_texts, candidate_texts):
    bert_scores_sum = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    num_samples = 0

    for ref_text, cand_text in zip(reference_texts, candidate_texts):
        if ref_text is not None and cand_text is not None:
            P, R, F1 = score([ref_text], [cand_text], lang='en', verbose=False)
            bert_scores_sum['precision'] += P.mean().item()
            bert_scores_sum['recall'] += R.mean().item()
            bert_scores_sum['f1'] += F1.mean().item()
            num_samples += 1

    if num_samples > 0:
        for metric in bert_scores_sum:
            bert_scores_sum[metric] /= num_samples

    return bert_scores_sum

def calculate_hint(reference_texts, generated_texts):
    hint_precisions, hint_hit_key_nums, hint_key_nums = [], [], []
    hint_recalls, hint_hit_word_nums, hint_word_nums = [], [], []
    for reference_text, generated_text in zip(reference_texts, generated_texts):
        hit_pred = []
        generated_text = generated_text.split(';')
        for index, text in enumerate(generated_text):
            text = text.strip()
            if text not in " ".join(hit_pred) and text in reference_text and text not in avoid_phrases:
                hit_pred.append(text)
        # calculate hit score and precision
        n = len(generated_text)
        hit_num = len(hit_pred)
        hit_precision = hit_num / n if n > 0 else 0

        hit_pred_word = []
        for text in " ".join(generated_text).split():
            text = text.strip()
            if text not in " ".join(hit_pred_word) and text in reference_text and text not in avoid_phrases:
                hit_pred_word.append(text)
        m = len(reference_text.split())
        hit_word_num = len(hit_pred_word)
        hit_word_recall = hit_word_num / m if m > 0 else 0

        hint_precisions.append(hit_precision)
        hint_hit_key_nums.append(hit_num)
        hint_key_nums.append(n)

        hint_recalls.append(hit_word_recall)
        hint_hit_word_nums.append(hit_word_num)
        hint_word_nums.append(m)

    result = {"hint_hit_key_num": np.mean(hint_hit_key_nums), "hint_key_precision": np.mean(hint_precisions),
              "key_num": np.mean(hint_key_nums),
              "hint_hit_word_num": np.mean(hint_hit_word_nums), "hint_word_recall": np.mean(hint_recalls),
              "word_num": np.mean(hint_word_nums)}

    return {k: round(v, 4) for k, v in result.items()}


###################prompt+article+RL2##########################################
# with open('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/3/RL-article-nep/1-0/0/epoch_4_val_split_predictions.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# # 获取特定字段的值
# all_gpt3_generated_texts = []
# all_ref_texts = []
# all_generated_texts = []
# # 循环遍历数据，获取每个字典中的 'generated_text' 字段的值，并添加到列表中
# ref_ix = 0
# for item in data:
#     # 检查 'gpt3_generated_text' 字段是否存在且不是空列表
#     if 'gpt3_generated_text' in item and item['gpt3_generated_text']:
#         gpt3_generated_text = item['gpt3_generated_text'][0]
#     else:
#         # 如果不存在或为空列表，则将 gpt3_generated_text 设为 None 或者其他默认值
#         gpt3_generated_text = None
#     #gpt3_generated_text = item['gpt3_generated_text'][0]
#     ref_text = item['ref_text']
#     generated_text = item['generated_text']
#     ref_text = ref_text.strip(f"<START-{ref_ix + 1}>").strip(f"<END-{ref_ix + 1}>")
#     all_gpt3_generated_texts.append(gpt3_generated_text)
#     all_ref_texts.append(ref_text)
#     all_generated_texts.append(generated_text)
#     ref_ix += 1
#
# # print("all_gpt3_generated_texts:", all_gpt3_generated_texts)
# # print("--------------------------------")
# # print("all_ref_texts:", all_ref_texts)
#
# print('Start evaluating ROUGE score and BERT score !!!')
# # Calculate ROUGE scores
# rouge_scores = calculate_rouge2(all_ref_texts, all_gpt3_generated_texts)
# print("ROUGE scores:", rouge_scores)
#
# # Calculate BERT score
# bert_scores = calculate_bert_score2(all_ref_texts, all_gpt3_generated_texts)
# print("BERT scores:", bert_scores)

# Calculate hint score
# hint_scores = calculate_hint(all_ref_texts, all_generated_texts)
# print("Hint scores:", hint_scores)



###################article####################模型==GPT3############################
# #读取 JSON 文件 /Users/dingjunmei/Desktop/article_SFT.json、/Users/dingjunmei/Desktop/article.json
# with open('/Users/dingjunmei/Desktop/article_SFT.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
# # 获取特定字段的值
# all_gpt3_generated_texts = []
# all_ref_texts = []
# for item in data:
#     gpt3_generated_text = item['gpt3_gen_texts'][0]
#     ref_text = item['summary']
#
#     all_gpt3_generated_texts.append(gpt3_generated_text)
#     all_ref_texts.append(ref_text)
#
# print('Start evaluating ROUGE score and BERT score !!!')
# # Calculate ROUGE scores
# rouge_scores = calculate_rouge(all_ref_texts, all_gpt3_generated_texts)
# print("ROUGE scores:", rouge_scores)
#
# # Calculate BERT score
# bert_scores = calculate_bert_score(all_ref_texts, all_gpt3_generated_texts)
# print("BERT scores:", bert_scores)



###################标注pre_keyword+article################模型==GPT3################################
# # 读取 JSON 文件
# with open('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/attack/RL2/1_1_1_val_attack.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
# # 获取特定字段的值
# all_gpt3_generated_texts = []
# all_ref_texts = []
# all_generated_texts = []
# for item in data:
#     gpt3_generated_text = item['gpt3_gen_texts'][0]
#     ref_text = item['summary']
#     generated_text = item['prediction']
#
#     all_gpt3_generated_texts.append(gpt3_generated_text)
#     all_ref_texts.append(ref_text)
#     all_generated_texts.append(generated_text)
#
# print('Start evaluating ROUGE score and BERT score !!!')
# # Calculate ROUGE scores
# rouge_scores = calculate_rouge(all_ref_texts, all_gpt3_generated_texts)
# print("ROUGE scores:", rouge_scores)
#
# # Calculate BERT score
# bert_scores = calculate_bert_score(all_ref_texts, all_gpt3_generated_texts)
# print("BERT scores:", bert_scores)
#
# Calculate hint score
# hint_scores = calculate_hint(all_ref_texts, all_generated_texts)
# print("Hint scores:", hint_scores)


# ###################prompt+article+RL1##########################################新生成
# with open('/Users/dingjunmei/Downloads/0-1-0_general_test.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
# # 获取特定字段的值
# all_gpt3_generated_texts = []
# all_ref_texts = []
# all_generated_texts = []
# # 循环遍历数据，获取每个字典中的 'generated_text' 字段的值，并添加到列表中
# ref_ix = 0
# for item in data:
#     gpt3_generated_text = item['gpt3_gen_texts2'][0]
#     ref_text = item['ref_text']
#     generated_text = item['generated_text']
#     ref_text = ref_text.strip(f"<START-{ref_ix + 1}>").strip(f"<END-{ref_ix + 1}>")
#     all_gpt3_generated_texts.append(gpt3_generated_text)
#     all_ref_texts.append(ref_text)
#     all_generated_texts.append(generated_text)
#     ref_ix += 1
#
# print('Start evaluating ROUGE score and BERT score !!!')
# # Calculate ROUGE scores
# rouge_scores = calculate_rouge(all_ref_texts, all_gpt3_generated_texts)
# print("ROUGE scores:", rouge_scores)
#
# # Calculate BERT score
# bert_scores = calculate_bert_score(all_ref_texts, all_gpt3_generated_texts)
# print("BERT scores:", bert_scores)
#
# # Calculate hint score
# hint_scores = calculate_hint(all_ref_texts, all_generated_texts)
# print("Hint scores:", hint_scores)


# # 读取 JSON 文件
with open('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/general/length/512-512_general_valid.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 获取特定字段的值
all_gpt3_generated_texts = []
all_ref_texts = []
all_generated_texts = []
# 循环遍历数据，获取每个字典中的 'generated_text' 字段的值，并添加到列表中
ref_ix = 0
for item in data:
    gpt3_generated_text = item['gpt3_generated_text'][0]
    ref_text = item['ref_text']
    generated_text = item['generated_text']
    ref_text = ref_text.strip(f"<START-{ref_ix + 1}>").strip(f"<END-{ref_ix + 1}>")
    all_gpt3_generated_texts.append(gpt3_generated_text)
    all_ref_texts.append(ref_text)
    all_generated_texts.append(generated_text)
    ref_ix += 1

print('Start evaluating ROUGE score and BERT score !!!')
# Calculate ROUGE scores
rouge_scores = calculate_rouge(all_ref_texts, all_gpt3_generated_texts)
print("ROUGE scores:", rouge_scores)

# # Calculate BERT score
# bert_scores = calculate_bert_score(all_ref_texts, all_gpt3_generated_texts)
# print("BERT scores:", bert_scores)
#
# # Calculate hint score
# hint_scores = calculate_hint(all_ref_texts, all_generated_texts)
# print("Hint scores:", hint_scores)

#gpt3_generated_text

##       处理llama2数据    ##
# # 读取 JSON 文件
# with open('/Users/dingjunmei/Downloads/test_article.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
# # 获取特定字段的值
# all_gpt3_generated_texts = []
# all_ref_texts = []
# # 循环遍历数据，获取每个字典中的 'generated_text' 字段的值，并添加到列表中
# ref_ix = 0
# for item in data:
#     gpt3_generated_text = item['vicuna_gen_texts']
#     ref_text = item['summary']
#     #generated_text = item['generated_text']
#
#     # 定义正则表达式模式，匹配以<s>开头和结尾的字符串
#     pattern = r'<s>.*?<s>'
#     # 使用正则表达式替换匹配到的部分为空字符串
#     gpt3_generated_text = re.sub(pattern, '', gpt3_generated_text, flags=re.DOTALL)
#
#     all_gpt3_generated_texts.append(gpt3_generated_text)
#     all_ref_texts.append(ref_text)
#     ref_ix += 1
#
# print('Start evaluating ROUGE score and BERT score !!!')
# # Calculate ROUGE scores
# rouge_scores = calculate_rouge(all_ref_texts, all_gpt3_generated_texts)
# print("ROUGE scores:", rouge_scores)
#
# # Calculate BERT score
# bert_scores = calculate_bert_score(all_ref_texts, all_gpt3_generated_texts)
# print("BERT scores:", bert_scores)





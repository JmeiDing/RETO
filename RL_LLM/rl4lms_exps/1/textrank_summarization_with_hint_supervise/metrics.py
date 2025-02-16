from rouge import Rouge
from bert_score import score
import json
import nltk
from rouge_score import rouge_scorer
import numpy as np

avoid_phrases = ["one", "two", "three", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "an", "the", "he", "she", "i", "we", "you", "they",
                 "it", "this", "that", "those", "these", "me", "them", "him", "his", "her", "my", "your", "its", "ours", "our", "their", "what",
                 "which", "who", "why", "when", "whom", "whose", "could", "with", " ", ",", ".", "?", "!", ";","-","_","="]
from rouge_score import rouge_scorer

def calculate_rouge(reference_texts_list, generated_texts_list):
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
    num_samples = len(reference_texts_list)

    for reference_texts, generated_texts in zip(reference_texts_list, generated_texts_list):
        scores = rouge_scorer_instance.score(generated_texts, reference_texts)
        for metric in scores:
            rouge_scores_sum[metric] += scores[metric].fmeasure

        # Calculate ROUGE-Lsum
        rougeLsum = (2 * scores['rougeL'].precision * scores['rougeL'].recall) / (scores['rougeL'].precision + scores['rougeL'].recall + 1e-12)
        rouge_scores_sum['rougeLsum'] += rougeLsum

    # Calculate average ROUGE scores
    for metric in rouge_scores_sum:
        rouge_scores_sum[metric] /= num_samples

    return rouge_scores_sum


def calculate_bert_score(reference_texts, candidate_texts):
    P, R, F1 = score(reference_texts, candidate_texts, lang='en', verbose=False)
    bert_scores = {'precision': round(P.mean().item(), 4), 'recall': round(R.mean().item(), 4), 'f1': round(F1.mean().item(), 4)}
    return bert_scores


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

        # print('---预测关键词数量---')
        # print('预测关键词', generated_text)
        # print('命中关键词',hit_pred)


        hit_pred_word = []
        for text in " ".join(generated_text).split():
            text = text.strip()
            if text not in " ".join(hit_pred_word) and text in reference_text and text not in avoid_phrases:
                hit_pred_word.append(text)
        m = len(reference_text.split())
        hit_word_num = len(hit_pred_word)
        hit_word_recall = hit_word_num / m if m > 0 else 0

        # print('---预测单词数量---')
        # print('预测单词', reference_text.split())
        # print('命中单词', hit_pred_word)

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




###################标注prompt+article##########################################
######模型==GPT3
# 读取 JSON 文件
with open('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/valid_train_100.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# # 获取特定字段的值
all_pred_text = []
all_ref_texts = []
all_target_text = []
for item in data:
    pred_text = item['Prediction']
    ref_text = item['summary']
    target_text = item['target']

    all_pred_text.append(pred_text)
    all_ref_texts.append(ref_text)
    all_target_text.append(target_text)
#
# Calculate hint score
# hint_scores = calculate_hint(all_ref_texts, all_target_text)
# print("Hint scores:", hint_scores)
hint_scores = calculate_hint(all_pred_text, all_target_text)
print("pred_text_Hint scores:", hint_scores)


###################article##########################################
######模型==GPT3
# 读取 JSON 文件
with open('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/train_valid_nokey.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 获取特定字段的值
all_gpt3_generated_texts = []
all_summart_texts = []
all_target_text = []
for item in data:
    gpt3_generated_texts = item['Prediction']
    summart_texts = item['summary']
    target_text = item['target']

    all_gpt3_generated_texts.append(gpt3_generated_texts)
    all_summart_texts.append(summart_texts)
    all_target_text.append(target_text)

# hint_scores = calculate_hint(all_summart_texts, all_target_text)
# print("Hint scores:", hint_scores)
hint_scores = calculate_hint(all_gpt3_generated_texts, all_target_text)
print("pred_text_Hint scores:", hint_scores)




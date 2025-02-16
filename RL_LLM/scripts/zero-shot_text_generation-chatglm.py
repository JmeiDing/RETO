from transformers import AutoTokenizer, AutoModel
from typing import List
import pandas as pd
import numpy as np
import json

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/chatglm-6b-int4",
                                          trust_remote_code=True)
model = AutoModel.from_pretrained("/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/chatglm-6b-int4",
                                  trust_remote_code=True).half().cuda()
model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# #你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)


# # 读取csv 输入==article
# data_frame = pd.read_csv('/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/data/attack/valid.csv')
# result_list = []
# num =0
# for index, row in data_frame.iterrows():
#     article = row['article']
#     article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
#     article = tokenizer.decode(article_ids)

#     input_text  = 'Summarize the article into a coherent and complete abstract.' + article
#     #input_text  = 'Extract the attack process of the article into a coherent and complete abstract.' + article

#     response, history = model.chat(tokenizer, input_text, history=[])
#     row['chatglm_gen_texts'] = response
#     print('完成～', response)
#     print('完成～'+ str(num))
#     num += 1
#     result_dict = {'article': row['article'], 'summary': row['summary'], 'chatglm_gen_texts' :row['chatglm_gen_texts']}
#     result_list.append(result_dict)

# # 将结果列表转换为 JSON 格式并写入文件
# json_file_path = "/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/data/attack/valid_article.json"
# with open(json_file_path, 'w') as json_file:
#     json.dump(result_list, json_file, indent=4)


# 读取csv 输入==article+target
data_frame = pd.read_csv('/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/data/general/valid.csv')
result_list = []
num = 0
for index, row in data_frame.iterrows():
    article = row['article']
    article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
    article = tokenizer.decode(article_ids)

    target = row['prediction']

    input_text = 'Summarize the article into a coherent and complete abstract that accurately incorporates the provided keywords. ' + target + 'The article is as follows. ' + article
    # input_text = 'Extract the attack process of the article into a coherent and complete abstract that accurately incorporates the provided keywords. ' + target + 'The article is as follows. ' + article

    response, history = model.chat(tokenizer, input_text, history=[])
    row['chatglm_gen_texts'] = response
    print('完成～', response)
    print('完成～' + str(num))
    num += 1
    result_dict = {'article': row['article'], 'prediction': row['prediction'], 'summary': row['summary'],
                   'chatglm_gen_texts': row['chatglm_gen_texts']}

    result_list.append(result_dict)

# 将结果列表转换为 JSON 格式并写入文件
json_file_path = "/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/data/general/valid_keyword_article.json"
with open(json_file_path, 'w') as json_file:
    json.dump(result_list, json_file, indent=4)


from transformers import AutoTokenizer, AutoModel
from typing import List
import pandas as pd
import numpy as np
import json

# tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/chatglm-6b-int4",
#                                           trust_remote_code=True)
# model = AutoModel.from_pretrained("/root/autodl-tmp/ChatGLM-6B/ChatGLM-6B-main/chatglm-6b-int4",
#                                   trust_remote_code=True).half().cuda()
# model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# #你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Llama-2-7b-hf").to("cuda")
model = model.eval()

# 读取csv 输入==article+target
data_frame = pd.read_csv('/root/autodl-tmp/llama-main/data/attack/valid.csv')
result_list = []
num = 0
for index, row in data_frame.iterrows():
    article = row['article']
    article_ids = tokenizer.encode(article, max_length=1024, truncation=True)
    article = tokenizer.decode(article_ids)

    target = row['prediction']

    # input_text = 'Summarize the article into a coherent and complete abstract that incorporates the provided keywords. ' + target + 'The article is as follows. ' + article
    input_text = 'Extract the attack process of the article into a coherent and complete abstract that incorporates the provided keywords. ' + target + 'The article is as follows. ' + article

    # response, history = model.chat(tokenizer, input_text, history=[])

    # print('input_text～' + input_text)
    # print('～～～～')
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(inputs["input_ids"], max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.1)
    output = output[0].to("cpu")
    # print(tokenizer.decode(output))
    row['llama_gen_texts'] = tokenizer.decode(output)
    print('完成～', row['llama_gen_texts'])
    print('完成～' + str(num))
    num += 1
    result_dict = {'article': row['article'], 'prediction': row['prediction'], 'summary': row['summary'],
                   'llama_gen_texts': row['llama_gen_texts']}
    result_list.append(result_dict)

# 将结果列表转换为 JSON 格式并写入文件
json_file_path = "/root/autodl-tmp/llama-main/data/attack/valid_keyword_article.json"
with open(json_file_path, 'w') as json_file:
    json.dump(result_list, json_file, indent=4)
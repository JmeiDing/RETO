from typing import Any, Dict, List
import os
import time
import openai
import logging
import random
from transformers import GPT2TokenizerFast
import requests


# 设置 OPENAI_API_KEY 环境变量，这两个变量是全局变量，不能放在里面
#openai.api_base = "https://one.aiskt.com/v1"
openai.api_base = "https://cd.aiskt.com/v1"
#openai.api_base = "https://key.aiskt.com/v1"
openai.api_key = "sk-kF7l8GkGRstUD0Xe7eEf44Fd92794e24B49a778868F60522"

#Claude 3
api_url = "http://one.aiskt.com/v1/messages"
api_key = "sk-kF7l8GkGRstUD0Xe7eEf44Fd92794e24B49a778868F60522"


# api_key = "sk-kF7l8GkGRstUD0Xe7eEf44Fd92794e24B49a778868F60522"
# api_base = "https://one.aiskt.com/v1"

# avoid_keywords = ["one", "two", "three", "1", "2", "3", "a", "he", "she", "i", "we", "you", "it", "this",
#         "that", "the", "those", "these", "they", "me", "them", "what", "him", "her", "my", "which", "who", "why",
#         "your", "my", "his", "her", "ours", "our", "could", "with", "whom", "whose"]

avoid_keywords = ["one", "two", "three", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "an", "the",
                  "he", "she", "i", "we", "you", "they", "it", "this", "that", "those", "these", "me",
                  "them", "him", "his", "her", "my", "your", "its", "ours", "our", "their", "what", "which",
                  "who", "why", "when", "whom", "whose", "could", "with"]

class GPT3():
    #  gpt-3.5-turbo-16k的max_prompt_length=16384
    def __init__(self, model="gpt-3.5-turbo-16k", interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4, max_prompt_length=16384):
        self.model = model
        self.interval = interval
        self.timeout = timeout
        self.base_timeout = timeout
        self.patience = patience
        self.exp = exp
        #self.max_prompt_length表述输入(prompt+源文档)和输出(摘要)的长度和，gpt-3.5-turbo最大为4096
        self.max_prompt_length = max_prompt_length
        self.max_interval = max_interval
        self.tokenizer = GPT2TokenizerFast.from_pretrained("./pretrain_model/gpt2")


    def call(self, prompt, temperature=1.0, top_p=1.0, max_tokens=64, n=1,
        frequency_penalty=0, presence_penalty=0, stop=["Q:"], rstrip=False,**kwargs):

        #openai.api_key = os.environ.get('OPENAI_API_KEY', None)
        # openai.api_base = "https://one.aiskt.com/v1"
        # openai.api_key = "sk - kF7l8GkGRstUD0Xe7eEf44Fd92794e24B49a778868F60522"


        # check if exceeding len limit
        # 这一行首先对输入的提示（prompt）进行 tokenization（分词处理），然后计算 tokenized 输入的长度。
        # 在这里，self.tokenizer 是一个分词器，它将提示转换为模型能够理解的 token 序列。
        # input_ids 是分词后生成的 token 序列。

        # prompt =gpt3_input_text，是general_hint_fs.txt样本的样子
        # Article: input_text(=airtcle)
        # Q: Write a short summary of the article in 2-4 sentences that accurately incorporates the provided keywords.
        # Keywords: generated_texts
        # A:

        # max_tokens生成的样本，在metrics中 max_tokens: 160
        # input = 4096 - 160 = 3936
        input_len = len(self.tokenizer(prompt).input_ids)
        if input_len + max_tokens >= self.max_prompt_length:
            logging.warning("OpenAI length limit error.")
            return [""] * n

        # stop words
        # 这段代码的目的是确保 stop 是一个列表，以便后续的代码可以在 stop 上执行期望的列表操作。
        # 如果 stop 是字符串，它将被转换为包含该字符串的单元素列表。
        if isinstance(stop, List):
            pass
        elif isinstance(stop, str):
            stop = [stop]

        if rstrip:
            # 是Python 字符串方法，用于去除字符串末尾（右侧）的空白字符，默认情况下是空格、制表符和换行符。
            # 这个方法并不会修改原始字符串，而是返回一个新的字符串。
            prompt = prompt.rstrip()

        retry_interval_exp = 1 
        t1 = time.time()

        while True and retry_interval_exp <= self.patience:
            try:
                if self.model == "gpt-3.5-turbo-16k": # chat completion
                    messages = [{"role": "user", "content": prompt}]
                    response = openai.ChatCompletion.create(model=self.model,
                                                        messages=messages,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,#生成文本的最大长度
                                                        n=n, #生成的文本数量
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop,
                                                        request_timeout=self.timeout # timeout!
                                                        )
                    candidates = response["choices"]
                    candidates = [candidate["message"]["content"] for candidate in candidates]

                elif self.model == "claude-3-opus-20240229": #
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": "claude-3-opus-20240229",  # 或其他模型名称
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                    }
                    # 发送POST请求
                    response = requests.post(api_url, headers=headers, json=data)
                    # 检查响应状态
                    if response.status_code == 200:
                        # 解析响应并提取需要的信息
                        candidates = response.json()
                        candidates = candidates['choices'][0]['message']['content']

                    else:
                        logging.info(f'Error: Received status code {response.status_code}')




                else: # text completion openai.ChatCompletion.create
                    response = openai.Completion.create(model=self.model,
                                                        prompt=prompt,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        n=n,
                                                        top_p=top_p,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop,
                                                        request_timeout=self.timeout # timeout!
                                                        )    
                    candidates = response["choices"]
                    candidates = [candidate["text"] for candidate in candidates]
                
                t2 = time.time()
                logging.info(f"{input_len} tokens, {t2-t1} secs")  

                return candidates

            # except openai.error.RateLimitError as e:
            except Exception as e:
                # logging.warning("OpenAI rate limit error. Retry")
                logging.warning(e)
                # Exponential backoff
                time.sleep(max(self.max_interval, self.interval * (self.exp ** retry_interval_exp)))
                retry_interval_exp += 1
        
        return None
    

if __name__ == "__main__":
    gpt3 = GPT3()
    
    # messages = []
    # for i in range(100):
    #     messages.append(f"what is the sum of {random.randint(1000, 10000)} and {random.randint(1000, 10000)}?")
    # predictions = gpt3.async_call(prompt=messages)

    for i in range(10):
        message = f"what is the sum of {random.randint(1000, 10000)} and {random.randint(1000, 10000)}?"
        predictions = gpt3.call(prompt=message)
        print(message, predictions)
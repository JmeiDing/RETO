import pandas as pd

# 加载.parquet文件
data = pd.read_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/attack_parquet/valid.parquet')
print(data)

sample_count = len(data)
print("样本数量：", sample_count)

# # 保存为CSV文件
# data.to_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/attack_parquet/test.csv', index=False)
#
# # 读取CSV文件
# test = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/attack_parquet/test.csv', encoding='ISO-8859-1')
# print(test)


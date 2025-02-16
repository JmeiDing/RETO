import pandas as pd
import numpy as np
import csv


#general
# raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/general/raw_general_csv/train.csv')
# raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/general/raw_general_csv/test.csv')
# raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/general/raw_general_csv/valid.csv')

# len(general_train)=1077\len(general_test)=133\len(general_valid)=135

#attack
# raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/raw_attack_csv/train.csv')
# raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/raw_attack_csv/test.csv')
# raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/raw_attack_csv/valid.csv')

# len(general_train)=810\len(general_test)=103\len(general_valid)=101

#将csv转化为parquet；列名转化，增加id列
def csv_to_parquet(path):
    raw_csv = pd.read_csv(path)

    # 交换两列的顺序并创建新的DataFrame
    column1_name = 'summary'  # 替换为实际的列名
    column2_name = 'text'  # 替换为实际的列名
    # 创建新的DataFrame
    new_csv = raw_csv[[column2_name, column1_name]]
    #print(new_csv)

    # 增加列名
    new_csv['id'] = 'test_' + pd.Series(range(1, len(new_csv) + 1)).astype(str)
    #print(new_csv)

    # 更换列名
    new_column_names = {'text': 'article', 'summary': 'highlights'}
    raw_parquet = new_csv.rename(columns=new_column_names)
    print(raw_parquet)

    raw_parquet.to_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/general/general_parquet/test.parquet')

# csv_to_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/general/raw_general_csv/test.csv')
# df = pd.read_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/general/general_parquet/test.parquet')
# print(df)

#从parquet采样
def sample_parquet(path):
    df = pd.read_parquet(path)
    # train:3；test:2；valid：2
    sampled_df = df.sample(n=2)
    print(sampled_df)
    sampled_df.to_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/sample_attack/valid.parquet')

# sample_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/attack_parquet/test.parquet')
# df = pd.read_parquet('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/attack_parquet/test.parquet')
# print(df)


#parquet_to_csv
# 加载 .npy 文件
# npy_file = "/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/textrank-all/valid.npy"
# data = np.load(npy_file, allow_pickle=True)
#
# # 跳过第一行
# data = data[1:]
# # 获取字段名（字典的键）
# fields = list(data[0].keys())
#
# # 打开一个新的 .csv 文件并写入数据
# csv_file = "/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/textrank-all_csv/valid.csv"
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#
#     # 写入字段名
#     writer.writerow(fields)
#
#     # 写入每一行数据
#     for item in data:
#         row = [item[field] for field in fields]
#         writer.writerow(row)
#
# print(f"Data successfully written to {csv_file}")

#构造bart数据
raw_csv = pd.read_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/attack_csv/valid.csv')

article = raw_csv.iloc[:, 4] + raw_csv.iloc[:, 0]
summary = raw_csv.iloc[:, 1]
merged = pd.concat([summary,article], axis=1)
merged.columns = ['summary', 'text']
print(merged)
merged.to_csv('/Users/dingjunmei/code/RL_LLM/sft4lms/data/attack/super_csv/valid.csv', index=False)



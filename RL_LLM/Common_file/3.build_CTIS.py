import os
import pandas as pd

# 定义文件夹路径
folder1 = '/Users/dingjunmei/code/RL_LLM/Common_file/general_tgt'
folder2 = '/Users/dingjunmei/code/RL_LLM/Common_file/com_src'
output_csv = '/Users/dingjunmei/code/RL_LLM/Common_file/combine_file/CTIS.csv'

# 获取文件夹1和文件夹2中的文件名
files_in_folder1 = set(os.listdir(folder1))
files_in_folder2 = set(os.listdir(folder2))

# 找到同时存在于两个文件夹中的文件名
common_files = files_in_folder1.intersection(files_in_folder2)

# 创建一个空的DataFrame
df = pd.DataFrame(columns=['summary', 'text'])

# 读取文件内容并合并
for file_name in common_files:
    file_path1 = os.path.join(folder1, file_name)
    file_path2 = os.path.join(folder2, file_name)

    with open(file_path1, 'r', encoding='utf-8') as f1:
        summary_content = f1.read().strip()

    with open(file_path2, 'r', encoding='utf-8') as f2:
        text_content = f2.read().strip()

    # 将内容添加到DataFrame
    df = df.append({'summary': summary_content, 'text': text_content}, ignore_index=True)

# 将DataFrame保存为CSV文件
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"共找到 {len(common_files)} 个共同文件，并已将合并后的内容保存到 {output_csv} 文件中。")

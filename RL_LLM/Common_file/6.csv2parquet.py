import os
import pandas as pd


def convert_csv_to_parquet(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # 生成输出文件路径和文件名
            parquet_filename = os.path.splitext(filename)[0] + '.parquet'
            output_path = os.path.join(output_folder, parquet_filename)

            # 将数据框架转换为Parquet文件
            df.to_parquet(output_path)


# 指定输入和输出文件夹路径
input_folder = '/Users/dingjunmei/code/RL_LLM/Common_file/general_sample/general_csv'
output_folder = '/Users/dingjunmei/code/RL_LLM/Common_file/general_sample/general_parquet_new'

convert_csv_to_parquet(input_folder, output_folder)

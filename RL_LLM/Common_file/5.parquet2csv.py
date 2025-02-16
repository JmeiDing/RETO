import os
import pandas as pd


def convert_parquet_to_csv(parquet_folder, csv_folder):
    # 确保CSV文件夹存在
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    # 遍历Parquet文件夹中的所有文件
    for file_name in os.listdir(parquet_folder):
        if file_name.endswith('.parquet'):
            # 读取Parquet文件
            parquet_path = os.path.join(parquet_folder, file_name)
            df = pd.read_parquet(parquet_path)

            # 将DataFrame转换为CSV文件
            csv_file_name = file_name.replace('.parquet', '.csv')
            csv_path = os.path.join(csv_folder, csv_file_name)
            df.to_csv(csv_path, index=False)

            print(f"Converted {parquet_path} to {csv_path}")


# 定义Parquet文件夹和CSV文件夹的路径
parquet_folder = '/Users/dingjunmei/code/RL_LLM/Common_file/general_sample/general_parquet_new'
csv_folder = '/Users/dingjunmei/code/RL_LLM/Common_file/general_sample/general_parquet_new/general_csv'

# 执行转换
convert_parquet_to_csv(parquet_folder, csv_folder)

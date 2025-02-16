import os
import shutil

# 定义文件夹路径
folder1 = '/Users/dingjunmei/Desktop/用户行为分析4/模型数据/raw_data/A-finaldata/Attack_Sum/attack_src_filter'
folder2 = '/Users/dingjunmei/Desktop/用户行为分析4/模型数据/raw_data/A-finaldata/General_Sum/src_filter'
output_folder = '/Users/dingjunmei/code/RL_LLM/Common_file/com_src'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹1和文件夹2中的文件名
files_in_folder1 = set(os.listdir(folder1))
files_in_folder2 = set(os.listdir(folder2))

# 找到同时存在于两个文件夹中的文件名
common_files = files_in_folder1.intersection(files_in_folder2)

# 将同时存在的文件复制到新的文件夹中
for file_name in common_files:
    file_path1 = os.path.join(folder1, file_name)
    file_path2 = os.path.join(folder2, file_name)
    output_path = os.path.join(output_folder, file_name)

    # 如果新文件夹中不存在该文件则复制文件
    if not os.path.exists(output_path):
        shutil.copy(file_path1, output_folder)

print(f"共找到 {len(common_files)} 个共同文件，并已复制到 {output_folder} 文件夹中。")

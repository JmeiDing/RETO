import pandas as pd

# 读取 CSV 文件
# valid = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/valid2.csv')
#
# prediction_nokey = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_valid_nokey.csv')
# # prediction_100 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_100_valid.csv')
# # prediction_85 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_85_valid.csv')
# # prediction_70 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_70_valid.csv')
# # prediction_55 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_55_valid.csv')
# # prediction_40 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_40_valid.csv')
# # prediction_25 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_25_valid.csv')
# # prediction_10 = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/predictions_bart-large-cnn_train_10_valid.csv')
# #
#
# #Prediction,Reference, summary,target
#
#
# sorted_df1 = valid.sort_values(by='summary', key=lambda x: x.str.len())
# sorted_df2 = prediction_nokey.sort_values(by='summary', key=lambda x: x.str.len())
#
#
# # 重新索引以确保索引顺序一致
# sorted_df1.reset_index(drop=True, inplace=True)
# sorted_df2.reset_index(drop=True, inplace=True)
#
# # 横向合并两个 DataFrame
# merged_df = pd.concat([sorted_df1, sorted_df2], axis=1)
#
# # 打印合并后的 DataFrame
# merged_df.to_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/train_valid_nokey.csv', index=False)


# 执行完上一步再执行下面的代码
merged_df = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/train_valid_nokey.csv', encoding='latin-1')
# 将 DataFrame 转换为 JSON 格式
json_data = merged_df.to_json(orient='records')


# 将 JSON 数据写入文件
with open('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/textrank_summarization_with_hint_supervise/train_valid_nokey.json', 'w') as f:
    f.write(json_data)
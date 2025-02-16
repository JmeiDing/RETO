#原始test.json，包含article、summary，直接计算指标

#原始test.json,转test.csv,读取article、summary
#预测pretest.csv,读取：prediction,summary，
#合并上述文件：pre_test.json

#gene_test不用管
import pandas as pd
# 从JSON文件中读取数据
gene_data = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/general/data/gene_val.csv', encoding='ISO-8859-1')
# 选择需要的字段Reference Text,Prompt Text
article = gene_data['Prompt Text'].str.replace("Extract the keywords: ", "")
article = pd.DataFrame(article)
summary = gene_data['Reference Text']
gene = pd.concat([article, summary], axis=1)
gene.columns = ['article', 'reference']
print(gene)

pred_data = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/general/data/pre_valid.csv', encoding='ISO-8859-1')
# 选择需要的字段
pred_data = pred_data[['prediction', 'summary']]
print(pred_data)

if len(gene) == len(pred_data):
    merged = pd.concat([gene, pred_data], axis=1)
    print(merged)
    merged.to_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/general/data/valid.csv', index=False)
else:
    print("Error: The number of rows in raw_data and pred_data are not the same.")


# test_pred = pd.read_csv('/rl4lms_exps/3/article-nep/test_pred.csv', encoding='ISO-8859-1')
# test_gene = pd.read_csv('/rl4lms_exps/3/article-nep/test_generated_data.csv', encoding='ISO-8859-1')
# # test_pred: Prediction,summary
# # test: summary,phrases,target
# # test generate: Generated Text,summary,Prompt Text,Meta Info
#
# #step1: summary、Prediction[关键词]、Prompt Text ——————pred_summary[article+关键词生成]
# step1_key = test_pred.iloc[:, 0] #test_gene['Reference Text']
# step1_sum = test_pred.iloc[:, 1]
# print(step1_key)
# print(step1_sum)
# step2_key = test_gene.iloc[:, 0]
# step2_sum = test_gene.iloc[:, 1]
# print(step2_key)
# print(step2_sum)
#
# prompt_text = test_gene.iloc[:, 2]
# article = prompt_text.str.replace("Extract the keywords: ", "")
# article = pd.DataFrame(article)
# article = article.iloc[:, 0]
#
#
#
# merged = pd.concat([article, step1_key, step1_sum, step2_key, step2_sum], axis=1)
# merged.columns = ['article', 'step1_key','step1_sum','step2_key','step2_sum']
# # print(merged)
# merged.to_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/3/article-nep/merged.csv', index=False)
#
# print(merged['step1_sum'])
# print(merged['step2_sum'])

# article = test_pred.iloc[:, 0] + test.iloc[:, 1]
# summary = test.iloc[:, 0]
# merged = pd.concat([article, summary], axis=1)
# merged.columns = ['article', 'summary']
# print(merged)

# #Generated Text,Reference Text,Prompt Text,Meta Info
# # 获取article
# Prompt_Text = test_gene.iloc[:, 2]
# # 使用 Pandas 的 Series 对象
# text = Prompt_Text.str.replace("Extract the keywords: ", "")
# # 将 article 转换为 Pandas DataFrame，并指定列名为 'article'
# text = pd.DataFrame({'text': text})
# # print(text)
# article = test_gene['Generated Text'] + text['text']
# summary = test_gene['Reference Text']
#
# test = pd.read_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/2/step2_key+article/test.csv',encoding='ISO-8859-1')
# reference = test.iloc[:, 0]
# # 合并两个DataFrame
# merged_df = pd.concat([article, summary, reference], axis=1)
# merged_df.columns = ['article', 'summary','reference']
#
#
# article = merged_df['article']
# summary = merged_df['reference']
# merged = pd.concat([article, summary], axis=1)
# merged.columns = ['article', 'summary']
# print(merged)
# # 保存合并后的DataFrame到CSV文件
#merged.to_csv('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/2/step1_key+article/step1_key_article.csv', index=False)
#

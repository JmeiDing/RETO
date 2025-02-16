import json


def read_and_combine_json_fields(json_file_path, output_txt_file_path):
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # 初始化字段值列表
        combined_texts = []

        # 检查JSON根元素类型
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    target = item.get('prediction', '')
                    article = item.get('article', '')
                    combined_text = f"Summarize the article into a coherent and complete abstract that incorporates the provided keywords: {target} The article is as follows: {article}"
                    combined_texts.append(combined_text)
        elif isinstance(data, dict):
            target = data.get('prediction', '')
            article = data.get('article', '')
            combined_text = f"Extract the attack process of the article and incorporate the provided keywords: {target} The article is as follows: {article}"
            combined_texts.append(combined_text)
        else:
            print("The JSON root element is neither a dictionary nor a list.")
            return

        # 检查是否有组合后的文本
        if not combined_texts:
            print(f"Fields 'target' and 'article' not found in JSON file.")
            return

        # 将组合后的文本保存到TXT文件
        with open(output_txt_file_path, 'w', encoding='utf-8') as txt_file:
            for text in combined_texts:
                txt_file.write(text + '\n')

        print(f"Combined texts have been saved to '{output_txt_file_path}'")

    except FileNotFoundError:
        print(f"File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file '{json_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")


# 示例用法
json_file_path = '/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/general/SFT/test_keyword_article.json'
output_txt_file_path = '/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/general/TXT/general_test.txt'

read_and_combine_json_fields(json_file_path, output_txt_file_path)


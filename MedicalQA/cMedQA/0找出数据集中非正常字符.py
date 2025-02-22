import json
import re

# 读取JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 去除汉字、中文标点、英文大写、英文小写字母、英文标点符号以及所有数字
def remove_unwanted_characters(text):
    pattern = r'[A-Za-z0-9\u4e00-\u9fff，。！？；：“”、（）《》〈〉【】〔〕‘’“”\'\"!@#\$%\^&\*\(\)\[\]{};:\'",<>\.\?/\\|`\~\-_+=]'
    return re.sub(pattern, '', text)

# 递归处理JSON对象中的所有字符串，并删除空值
def process_json(data):
    if isinstance(data, dict):
        # 递归处理字典并删除空值
        processed_dict = {key: process_json(value) for key, value in data.items()}
        return {k: v for k, v in processed_dict.items() if v not in [None, '', [], {}]}
    elif isinstance(data, list):
        # 递归处理列表并删除空值
        processed_list = [process_json(item) for item in data]
        return [item for item in processed_list if item not in [None, '', [], {}]]
    elif isinstance(data, str):
        # 处理字符串并删除空字符
        cleaned_text = remove_unwanted_characters(data)
        return cleaned_text if cleaned_text else None
    else:
        return data

# 保存处理后的数据到新的JSON文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
if __name__ == "__main__":
    # 输入和输出文件路径
    input_file_path = 'D:/code/MedicalQA/cMedQA/datasets/cMedQA.json'
    output_file_path = 'D:/code/MedicalQA/cMedQA/datasets/找出数据集中年非正常字符.json'

    # 读取JSON文件
    data = load_json(input_file_path)

    # 处理JSON数据
    processed_data = process_json(data)

    # 保存处理后的数据到新文件
    save_json(processed_data, output_file_path)

    print(f"处理完成，结果已保存至 {output_file_path}")

import json

# 读取json文件路径和要保存txt文件的路径
json_file_path = r'D:/code/MedicalQA/cMedQA/datasets/cMedQA_test(标准json).json'
txt_file_path = r'D:/code/ROUGE/reference/cMedQA.txt'

# 打开json文件并读取内容
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 打开txt文件，准备将内容写入
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    # 假设data是一个列表，逐条遍历数据
    for item in data:
        # 获取每个item的ans_contents第一个元素，并写入到txt中
        if 'ans_contents' in item and item['ans_contents']:
            first_ans_content = item['ans_contents'][0]
            txt_file.write(first_ans_content + '\n')

print(f"所有数据的第一个ans_contents元素已写入到{txt_file_path}")

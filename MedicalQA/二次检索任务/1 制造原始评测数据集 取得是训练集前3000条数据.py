import json

# 读取JSON文件
with open('二次检索任务/数据集/1 原始数据.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取每个元素的ans_contents的第一条数据
extracted_data = []
for item in data:
    if 'ans_contents' in item and item['ans_contents']:
        extracted_data.append(item['ans_contents'][0])

# 将提取的数据保存到txt文件中
with open('二次检索任务/数据集/reference_3000.txt', 'w', encoding='utf-8') as file:
    for line in extracted_data:
        file.write(line + '\n')
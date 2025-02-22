import json

# 读取输入的JSON文件
input_file = r'D:\code\MedicalQA\二次检索任务\数据集\2 ES检索数据.json'

# 读取输入文件的数据
with open(input_file, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 初始化三个列表，用于存储ans_contents的第1、2、3条数据
ans_contents1 = []
ans_contents2 = []
ans_contents3 = []

# 遍历每一条数据，提取ans_contents的第一条、第二条和第三条数据
for item in data:
    if 'ans_contents' in item:
        # 提取第一个元素
        if len(item['ans_contents']) > 0:
            ans_contents1.append(item['ans_contents'][0])
        else:
            ans_contents1.append(None)  # 如果没有第一个元素，添加None
        
        # 提取第二个元素
        if len(item['ans_contents']) > 1:
            ans_contents2.append(item['ans_contents'][1])
        else:
            ans_contents2.append(None)  # 如果没有第二个元素，添加None

        # 提取第三个元素
        if len(item['ans_contents']) > 2:
            ans_contents3.append(item['ans_contents'][2])
        else:
            ans_contents3.append(None)  # 如果没有第三个元素，添加None

# 将ans_contents1保存到txt文件
output_file1 = r'D:\code\MedicalQA\二次检索任务\数据集\2 ES评估1.txt'
with open(output_file1, 'w', encoding='utf-8') as outfile1:
    for ans in ans_contents1:
        outfile1.write(str(ans) + '\n')

# 将ans_contents2保存到txt文件
output_file2 = r'D:\code\MedicalQA\二次检索任务\数据集\2 ES评估2.txt'
with open(output_file2, 'w', encoding='utf-8') as outfile2:
    for ans in ans_contents2:
        outfile2.write(str(ans) + '\n')

# 将ans_contents3保存到txt文件
output_file3 = r'D:\code\MedicalQA\二次检索任务\数据集\2 ES评估3.txt'
with open(output_file3, 'w', encoding='utf-8') as outfile3:
    for ans in ans_contents3:
        outfile3.write(str(ans) + '\n')

print("数据处理完成，已保存到对应的TXT文件。")

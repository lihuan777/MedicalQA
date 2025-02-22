import json
from collections import Counter

# 读取训练数据
train_file = '个性化分类任务/数据集/webMedQA/test_data_cleaned.json'

with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 提取所有医生类型并统计频次
all_doctors = []

for item in train_data:
    categories = item.get('categories', [])
    
    # 如果 categories 是列表且不为空，提取第一个医生类型
    if isinstance(categories, list) and categories:
        all_doctors.append(categories[0])  # 只取第一个数据

# 使用 Counter 统计每个医生类型的频次
doctor_counts = Counter(all_doctors)

# 将结果转换为列表并按频次从高到低排序
sorted_doctors = sorted(doctor_counts.items(), key=lambda x: x[1], reverse=True)

# 输出医生类型及其频次，仅输出频次大于10次的类型
print("医生类型及其出现频次（按频次从高到低，仅频次大于10次）：")
for doctor, count in sorted_doctors:
    if count > 10:  # 修改为只输出频次大于10次的类型
        print(f"{doctor}: {count}")



# 打开文件并读取数据
file_path = '个性化分类任务/数据集/webMedQA/medQA.test.txt'  # 替换为你的文件路径
department_count = {}  # 用于存储科室及其出现次数

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 假设每行数据用制表符分隔，科室信息在第一列
        department = line.strip().split('\t')[0]
        if department in department_count:
            department_count[department] += 1
        else:
            department_count[department] = 1

# 输出结果
total_departments = len(department_count)
print(f"总共有 {total_departments} 个科室。")
print("每个科室出现的次数如下：")
for department, count in department_count.items():
    print(f"{department}: {count}次")
import json
import random

# 加载数据集
with open('cMedQA/datasets/cMedQA.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 打乱数据，确保随机性
random.seed(42)

# 切分数据集：5000条作为测试集，其余作为训练集
test_data = data[:5000]
train_data = data[5000:]

# 保存测试集
with open('cMedQA/datasets/cMedQA_test.json', 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, ensure_ascii=False, indent=4)

# 保存训练集
with open('cMedQA/datasets/cMedQA_train.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

print("数据集切分完成，测试集保存在 cMedQA_test.json，训练集保存在 cMedQA_train.json")

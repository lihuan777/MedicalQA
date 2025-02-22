import json

# 假设数据已经读取到内存中
train_file = '个性化分类任务/数据集/山大数据集/train_data_cleaned.json'

with open(train_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 计算每个部分的长度
lengths = {
    "ques_title": [],
    "ques_content": [],
    "ans_contents": [],
    "categories": []
}

for item in data:
    lengths["ques_title"].append(len(item["ques_title"]))
    lengths["ques_content"].append(len(item["ques_content"]))
    lengths["ans_contents"].extend([len(ans) for ans in item["ans_contents"]])
    lengths["categories"].extend([len(cat) for cat in item["categories"]])

# 计算最长长度
max_lengths = {
    "ques_title": max(lengths["ques_title"]),
    "ques_content": max(lengths["ques_content"]),
    "ans_contents": max(lengths["ans_contents"]),
    "categories": max(lengths["categories"])
}

# 计算平均长度
avg_lengths = {
    "ques_title": sum(lengths["ques_title"]) / len(lengths["ques_title"]),
    "ques_content": sum(lengths["ques_content"]) / len(lengths["ques_content"]),
    "ans_contents": sum(lengths["ans_contents"]) / len(lengths["ans_contents"]),
    "categories": sum(lengths["categories"]) / len(lengths["categories"])
}

# 输出结果
print("最长长度:", max_lengths)
print("平均长度:", avg_lengths)
import json

# 需要保留的医生类型列表
doctor_types = [
    '内科', '妇产科', '外科', '耳鼻喉科', '皮肤科', 
    '儿科', '肿瘤学', '整形美容外科', '精神病学', 
    '性病学', '传染病'
]

# 读取训练数据
train_file = '个性化分类任务/数据集/train_data.json'

with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 使用列表推导式过滤掉不包含指定医生类型的条目
filtered_data = []

for item in train_data:
    categories = item.get('categories', [])
    
    # 如果 categories 是列表类型，检查是否包含任何匹配的医生类型
    if isinstance(categories, list):
        matching_doctor = next((doctor for doctor in categories if doctor in doctor_types), None)
        
        # 如果找到匹配的医生类型，保留该条数据
        if matching_doctor:
            # 只保留第一个匹配的医生类型
            item['categories'] = [matching_doctor]
            filtered_data.append(item)

# 保存处理后的数据到新文件
output_file = '个性化分类任务/数据集/train_data_cleaned.json'

print(len(filtered_data))

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"Data has been cleaned and saved to {output_file}")



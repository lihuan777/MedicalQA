import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




def predict(dataset_name, model_name):


    test_file = f'个性化分类任务/数据集/{dataset_name}/test_data_cleaned.json'
    train_file = '个性化分类任务/数据集/{dataset_name}/train_data_cleaned.json'
    model_file = f'个性化分类任务/结果/{model_name}/checkpoint-Huatuo-26M'
    # 保存预测结果到文件
    output_dir = f'个性化分类任务/结果/{model_name}'  # 生成输出目录
    output_file = output_dir + f'/{dataset_name}.json'


    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 将JSON数据转为DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # 合并 Q 和 A
    train_df['text'] = train_df['ques_title'] + " " + train_df['ques_content'] + " " + train_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')
    test_df['text'] = test_df['ques_title'] + " " + test_df['ques_content'] + " " + test_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')

    # 编码科室信息 C
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['categories'].apply(lambda x: x[0] if isinstance(x, list) else ''))
    test_df['label'] = label_encoder.transform(test_df['categories'].apply(lambda x: x[0] if isinstance(x, list) else ''))  # 使用训练集中的标签编码

    # 转换为datasets格式
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_file)

    # 数据预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    # 对测试集进行tokenize
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 设置数据格式
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 加载训练好的模型
    model = AutoModelForSequenceClassification.from_pretrained(model_file, num_labels=len(label_encoder.classes_))
    model.to(device)  # 将模型移到device

    # 预测函数：直接通过模型预测
    def predict(texts):
        # Tokenize输入文本
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        # 返回预测的标签
        return label_encoder.inverse_transform(predictions.cpu().numpy())

    # 对测试集进行预测
    test_texts = test_df['text'].tolist()
    predictions = predict(test_texts)

    # 保存预测结果到文件
    output_data = {
        'predictions': predictions.tolist(),
        'true_labels': test_df['label'].tolist(),
        'labels': label_encoder.classes_.tolist()
    }



    # 确保目录存在
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)


    # 将结果保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to: {output_file}")



dataset_name = '山大数据集'
model_name = 'BART'
# predict(dataset_name, model_name)

dataset_name = 'CDMM'
model_name = 'BART'
predict(dataset_name, model_name)

dataset_name = 'Huatuo-26M'
model_name = 'BART'
predict(dataset_name, model_name)


dataset_name = 'webMedQA'
model_name = 'BART'
predict(dataset_name, model_name)


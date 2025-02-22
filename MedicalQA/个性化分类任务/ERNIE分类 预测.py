import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
import os
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Import tqdm for progress bars


# 加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def predict(dataset_name, model_name):


    test_file = f'个性化分类任务/数据集/{dataset_name}/test_data_cleaned.json'
    model_file = f'个性化分类任务/结果/{model_name}/checkpoint-CMDD'
    # 保存预测结果到文件
    output_dir = f'个性化分类任务/结果/{model_name}'  # 生成输出目录
    output_file = output_dir + f'/{dataset_name}.json'

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 将JSON数据转为DataFrame
    test_df = pd.DataFrame(test_data)

    # 合并 Q 和 A
    test_df['text'] = test_df['ques_title'] + " " + test_df['ques_content'] + " " + test_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')

    # 编码科室信息 C
    label_encoder = LabelEncoder()
    test_df['label'] = label_encoder.fit_transform(test_df['categories'].apply(lambda x: x[0] if isinstance(x, list) else ''))

    print("Classes:", label_encoder.classes_)
    
    # 转换为datasets格式
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # 加载BERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_file)

    # 数据预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    # 对测试集进行tokenize
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 设置数据格式
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 加载已训练的BERT模型
    model = AutoModelForSequenceClassification.from_pretrained(model_file, num_labels=len(label_encoder.classes_))
    model.to(device)  # 将模型移到device

    # 预测函数
    def predict(texts):
        model.eval()
        predictions = []

        # 使用tqdm显示进度
        for text in tqdm(texts, desc="Predicting", unit="sample"):
            # Tokenize input text
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=-1).item()
                predictions.append(label_encoder.inverse_transform([predicted_label])[0])

        return predictions

    # 对测试集进行预测
    test_texts = test_df['text'].tolist()
    predictions = predict(test_texts)

    output_data = {
        'predictions': predictions,
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
model_name = 'ERNIE'
# predict(dataset_name, model_name)

dataset_name = 'CMDD'
model_name = 'ERNIE'
predict(dataset_name, model_name)

dataset_name = 'Huatuo-26M'
model_name = 'ERNIE'
predict(dataset_name, model_name)


dataset_name = 'webMedQA'
model_name = 'ERNIE'
predict(dataset_name, model_name)
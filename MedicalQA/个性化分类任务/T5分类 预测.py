import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
import pandas as pd
import json
import os

# 加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




def predict(dataset_name, model_name):


    test_file = f'个性化分类任务/数据集/{dataset_name}/test_data_cleaned.json'
    train_file = f'个性化分类任务/数据集/{dataset_name}/train_data_cleaned.json'
    model_file = f'个性化分类任务/结果/{model_name}/checkpoint-{dataset_name}'
    # 保存预测结果到文件
    output_dir = f'个性化分类任务/结果/{model_name}'  # 生成输出目录
    output_file = output_dir + f'/{dataset_name}.json'
        
    # 加载保存的模型和 Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_file)
    model = T5ForSequenceClassification.from_pretrained(model_file)
    model.to(device)

    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    train_df = pd.DataFrame(train_data)

    # 提取所有类别标签
    all_categories = train_df['categories'].apply(lambda x: x[0] if isinstance(x, list) else '').unique()

    # 重新生成 LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(all_categories)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_df = pd.DataFrame(test_data)
    test_df['text'] = test_df['ques_title'] + " " + test_df['ques_content'] + " " + test_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')

    # 数据预处理函数
    def preprocess_function(texts):
        inputs = ['分类：' + text for text in texts]
        return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

    # 逐批次预测
    batch_size = 16
    predictions = []

    model.eval()
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df['text'].iloc[i:i+batch_size].tolist()
        inputs = preprocess_function(batch_texts)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)

    # 将索引转换为类别标签
    predicted_labels = label_encoder.inverse_transform(predictions)

    # 保存预测结果
    output_data = {
        'predictions': predicted_labels.tolist(),
        'true_labels': test_df['label'].tolist() if 'label' in test_df else [],
        'labels': label_encoder.classes_.tolist()
    }


    # 确保目录存在
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)



    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to: {output_file}")





# dataset_name = '山大数据集'
# model_name = 'T5'
# predict(dataset_name, model_name)

dataset_name = 'CMDD'
model_name = 'T5'
predict(dataset_name, model_name)

dataset_name = 'Huatuo-26M'
model_name = 'T5'
predict(dataset_name, model_name)


dataset_name = 'webMedQA'
model_name = 'T5'
predict(dataset_name, model_name)

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd
from datetime import datetime
import os

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")










def run_train(dataset,):

    output_dir = '个性化分类任务/结果/BERT'
    output_file = output_dir + dataset + '.json'


    # 加载数据集（JSON格式）
    train_file = '个性化分类任务/数据集/' + dataset + '/train_data_cleaned.json'
    test_file = '个性化分类任务/数据集/' + dataset + '/test_data_cleaned.json'

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


    print("Classes:", label_encoder.classes_)




    # 转换为datasets格式
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # 加载BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 数据预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    # 对训练集和测试集进行tokenize
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 设置数据格式
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 加载BERT模型
    model = BertForSequenceClassification.from_pretrained('个性化分类任务/模型/models--bert-base-chinese', num_labels=len(label_encoder.classes_))
    model.to(device)  # 将模型移到device







    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./' + dataset + '_BERT',          # 输出目录
        num_train_epochs=3,             # 训练周期数
        per_device_train_batch_size=8,  # 每个设备的训练批次大小
        per_device_eval_batch_size=16,  # 每个设备的评估批次大小
        warmup_steps=500,               # 预热步数
        weight_decay=0.01,              # 权重衰减
        logging_dir='./logs',           # 日志目录
        logging_steps=10,
        evaluation_strategy="epoch",    # 每个周期评估一次
        save_strategy="epoch",          # 每个周期保存一次
    )

    # 使用Trainer进行训练
    trainer = Trainer(
        model=model,                         # 模型
        args=training_args,                  # 训练参数
        train_dataset=train_dataset,         # 训练集
        eval_dataset=test_dataset,           # 测试集
        tokenizer=tokenizer,                 # Tokenizer
    )

    # 训练模型
    trainer.train()

    # 在测试集上评估模型
    results = trainer.evaluate(test_dataset)
    print("Evaluation results:", results)

    # 预测函数
    def predict(texts):
        # Tokenize input text
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


    



dataset = '山大数据集'
run_train(dataset)

dataset = 'CMDD'
run_train(dataset)

dataset = 'Huatuo-26M'
run_train(dataset)

dataset = 'webMedQA'
run_train(dataset)





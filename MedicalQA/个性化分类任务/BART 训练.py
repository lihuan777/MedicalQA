import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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



def run_train(dataset):

    output_dir = '个性化分类任务/结果/BART'


    # 加载数据集（JSON格式）
    train_file = '个性化分类任务/数据集/' + dataset + '/train_data_cleaned.json'
    test_file = '个性化分类任务/数据集/' + dataset + '/test_data_cleaned.json'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

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
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # 加载 RoBERTa Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/MedicalQA/2_model/bart-large-chinese')

    # 数据预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    # 对训练集和测试集进行tokenize
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 设置数据格式
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 加载 RoBERTa 模型并添加分类头
    model = AutoModelForSequenceClassification.from_pretrained('/root/autodl-tmp/MedicalQA/2_model/bart-large-chinese', num_labels=len(label_encoder.classes_))
    model.to(device)  # 将模型移到device

    # 定义训练参数

    output_dir = f'autodl-tmp/checkpoint-{dataset}-BART'
    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = 3,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 8,
        warmup_steps = 500,
        weight_decay = 0.01,
        logging_dir = './logs',
        logging_steps = 10,
        evaluation_strategy = "epoch",
        save_strategy = "no",  # 关闭自动保存
        save_total_limit = 1,
        load_best_model_at_end = False
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


    # 保存最终模型和相关信息
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存标签编码器
    import joblib
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))


    # 在测试集上评估模型
    results = trainer.evaluate(test_dataset)
    print("Evaluation results:", results)




dataset = '山大数据集'
# run_train(dataset)

dataset = 'CMDD'
run_train(dataset)

dataset = 'Huatuo-26M'
run_train(dataset)

dataset = 'webMedQA'
run_train(dataset)


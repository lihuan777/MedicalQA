import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def predict(dataset_name, model_name):
    test_file = f'个性化分类任务/数据集/{dataset_name}/test_data_cleaned.json'
    model_file = f'个性化分类任务/模型/glm-4-9b-chat'
    output_dir = f'个性化分类任务/结果/{model_name}'
    output_file = output_dir + f'/{dataset_name}.json'

    os.makedirs(output_dir, exist_ok=True)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_df = pd.DataFrame(test_data)
    test_df['text'] = test_df['ques_title'] + " " + test_df['ques_content'] + " " + test_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')

    # 动态加载科室标签
    labels = test_df['categories'].unique().tolist()  # 更改为从label列获取唯一值
    labels = sorted(labels)  # 确保标签顺序一致

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_file, trust_remote_code=True).to(device)

    # 数据预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    test_dataset = test_df[['text']].apply(preprocess_function, axis=1).tolist()

    def predict(texts):
        model.eval()
        predictions = []
        labels_str = ", ".join(labels)  # 动态生成科室列表字符串

        for text in tqdm(texts, desc="Predicting", unit="text"):
            # 动态构建提示模板
            prompt = f"根据以下医疗问题与医生的回答，请根据问题的症状、诊断或治疗建议，选择最合适的医生科室。从以下{len(labels)}个科室中选择，并只输出科室名称：\n{labels_str}\n问题与医生回答：{text}\n请选择一个科室，并仅输出科室名称，不允许有其他输出，只能从给定选项中选择。"

            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt",
                                                 return_dict=True
                                                 ).to(device)
            
            with torch.no_grad():
                gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predicted_label = predicted_text.strip()
            # 严格检查预测结果是否在标签列表中
            if predicted_label not in labels:
                predicted_label = "其他" if "其他" in labels else labels[0]  # 后备策略
            
            predictions.append(predicted_label)
        
        return predictions

    test_texts = test_df['text'].tolist()
    predictions = predict(test_texts)

    output_data = {
        'predictions': predictions,
        'true_labels': test_df['label'].tolist(),
        'labels': labels
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to: {output_file}")

# 示例调用
datasets = ['山大数据集', 'CDMM', 'Huatuo-26M', 'webMedQA']
model_name = 'glm-4-9b-chat'

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    predict(dataset, model_name)
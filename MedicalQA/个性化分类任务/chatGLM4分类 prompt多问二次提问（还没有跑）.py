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

def predict(dataset, model_name):


    test_file = f'个性化分类任务/数据集/{dataset}/test_data_cleaned.json'
    model_file = f'个性化分类任务/模型/glm-4-9b-chat'
    output_dir = f'个性化分类任务/结果/{model_name}'
    output_file = os.path.join(output_dir, f'{dataset}.json')



    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 将JSON数据转为DataFrame
    test_df = pd.DataFrame(test_data)
    test_df['text'] = test_df['ques_title'] + " " + test_df['ques_content'] + " " + test_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')
    # 从数据中提取唯一科室类型
    labels = test_df['categories'].unique().tolist()  # 修改这里

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_file, trust_remote_code=True).to(device)

    # 改进后的预测函数
    def predict_with_retry(text):
        messages = [{
            "role": "user",
            "content": f"""根据以下医疗问题与医生的回答，请根据问题的症状、诊断或治疗建议，选择最合适的医生科室。
从以下六个科室中选择，并只输出科室名称：{', '.join(labels)}
问题与医生回答：{text}
请选择一个科室，并仅输出科室名称，不允许有其他输出，只能从六个选项中选择。"""
        }]
        
        max_retries = 3  # 最大重试次数
        for attempt in range(max_retries):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=1
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # 处理可能的重复输出
            response = response.split('\n')[0].strip()
            
            # 如果结果有效则返回
            if response in labels:
                return response
            else:
                # 添加对话历史和新提示
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"输出包含无效内容：{response}。请严格按照要求，只从指定科室中选择并直接输出科室名称。"
                })
        
        # 如果重试后仍无效则返回第一个标签
        return labels[0]

    # 批量预测
    predictions = []
    for text in tqdm(test_df['text'].tolist(), desc="Predicting"):
        pred = predict_with_retry(text)
        predictions.append(pred)

    # 保存结果
    output_data = {
        'predictions': predictions,
        'true_labels': test_df['label'].tolist(),
        'labels': labels
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Predictions saved to: {output_file}")

# 数据集配置
datasets = ['山大数据集', 'CMDD', 'Huatuo-26M', 'webMedQA']
model_name = 'glm-4-9b-chat'

for dataset in datasets:
    
    
    predict(dataset, model_name)
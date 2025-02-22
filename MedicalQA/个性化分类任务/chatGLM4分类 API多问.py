import json
import pandas as pd
import os
from tqdm import tqdm
import requests  # 用于调用 API
from zhipuai import ZhipuAI

# 设备设置

client = ZhipuAI(api_key="366109b4d0ea4c579bc6decfc0f7bb74.HxHy1IwoM2NtvEAD")  # 请填写您自己的APIKey


def predict_with_chatglm_api(dataset_name, model_name):
    test_file = f'个性化分类任务/数据集/{dataset_name}/test_data_cleaned.json'
    output_dir = f'个性化分类任务/结果/{model_name}'
    output_file = output_dir + f'/{dataset_name}.json'

    os.makedirs(output_dir, exist_ok=True)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_df = pd.DataFrame(test_data)
    test_df['text'] = test_df['ques_title'] + " " + test_df['ques_content'] + " " + test_df['ans_contents'].apply(lambda x: x[0] if isinstance(x, list) else '')

    # 动态加载科室标签
    test_df['categories'] = test_df['categories'].apply(lambda x: x if isinstance(x, str) else ', '.join(x))

    # 动态加载标签
    labels = test_df['categories'].unique().tolist()
    labels = sorted(labels)

    def predict(texts):

        predictions = []
        labels_str = ", ".join(labels)  # 动态生成科室列表字符串

        for text in tqdm(texts, desc="Predicting", unit="text"):
            # 动态构建提示模板
            prompt = f"根据以下医疗问题与医生的回答，请根据问题的症状、诊断或治疗建议，选择最合适的医生科室。从以下{len(labels)}个科室中选择，并只输出科室名称：\n{labels_str}\n问题与医生回答：{text}\n请选择一个科室，并仅输出科室名称，不允许有其他输出，只能从给定选项中选择。"

            

            


            try:
                response = client.chat.completions.create(
                    model="glm-4-plus",  # 请填写您要调用的模型名称
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                predicted_text = response.choices[0].message.content
                predicted_label = predicted_text.strip()
                # 严格检查预测结果是否在标签列表中
                if predicted_label not in labels:
                    predicted_label = "其他" if "其他" in labels else labels[0]  # 后备策略

                predictions.append(predicted_label)
            except Exception as e:
                print(f"API 调用失败: {e}")
                predictions.append("其他")  # 如果 API 调用失败，使用默认标签

        return predictions

    test_texts = test_df['text'].tolist()
    predictions = predict(test_texts)

    output_data = {
        'predictions': predictions,
        'labels': labels
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to: {output_file}")

# 示例调用
datasets = ['CMDD', 'Huatuo-26M', 'webMedQA']
model_name = 'chatGLM-API'

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    predict_with_chatglm_api(dataset, model_name)


    
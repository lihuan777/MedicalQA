import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd
import os
from tqdm import tqdm
import re

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MedicalDepartmentClassifier:
    def __init__(self, bert_model_path, chatglm_model_path=None, confidence_threshold=0.8):
        # 初始化BERT模型
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
        self.bert_model.to(device)
        
        # 初始化标签编码器
        self.label_encoder = LabelEncoder()
        
        # 初始化大模型
        self.chatglm_tokenizer = None
        self.chatglm_model = None
        if chatglm_model_path:
            self._init_chatglm(chatglm_model_path)
            
        # 配置参数
        self.confidence_threshold = confidence_threshold
        self.department_list = []  # 将在训练时初始化

    def _init_chatglm(self, model_path):
        """初始化ChatGLM模型"""
        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.chatglm_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).half().cuda()  # 使用半精度节省显存

    def _bert_predict(self, text):
        """BERT单条预测并返回置信度"""
        inputs = self.bert_tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            top_prob, pred_label = torch.max(probs, dim=-1)
            
        return {
            "label": pred_label.item(),
            "confidence": top_prob.item(),
            "probs": probs.cpu().numpy()[0]
        }

    def _chatglm_correct(self, text, original_pred, confidence):
        """调用大模型进行修正"""
        prompt = f"""【医疗文本分类修正任务】
当前BERT模型将以下患者描述分类为：{original_pred}（置信度：{confidence:.2f}）
请仔细分析文本内容，从以下科室列表中选择最合适的科室：
{", ".join(self.department_list)}

患者描述：
{text}

请严格按照JSON格式返回结果：
{{
  "final_department": "科室名称",
  "reason": "修正理由（如无需修正请注明'维持原分类'）"
}}"""
        
        try:
            response, _ = self.chatglm_model.chat(
                self.chatglm_tokenizer,
                prompt,
                temperature=0.1,  # 降低随机性
                max_length=512
            )
            
            # 提取JSON内容
            json_str = re.search(r'{.*}', response, re.DOTALL).group()
            result = json.loads(json_str)
            
            # 有效性校验
            if result["final_department"] in self.department_list:
                return result
            return {"final_department": original_pred, "reason": "无效科室名称"}
        except Exception as e:
            print(f"修正失败: {str(e)}")
            return {"final_department": original_pred, "reason": "解析错误"}

    def predict_pipeline(self, test_file, output_dir):
        """完整预测流程"""
        # 加载测试数据
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        test_df = pd.DataFrame(test_data)
        test_df['text'] = test_df.apply(lambda x: 
            f"{x['ques_title']} {x['ques_content']} {x['ans_contents'][0] if isinstance(x['ans_contents'], list) else ''}", 
            axis=1
        )
        
        # 初始化标签编码器
        self.label_encoder.fit(test_df['categories'].apply(lambda x: x[0]))
        self.department_list = self.label_encoder.classes_.tolist()
        
        # 预测流程
        correction_logs = []
        final_predictions = []
        
        for text in tqdm(test_df['text'].tolist(), desc="Processing"):
            # BERT基础预测
            bert_result = self._bert_predict(text)
            original_label = self.label_encoder.inverse_transform([bert_result["label"]])[0]
            
            # 高置信度直接通过
            if bert_result["confidence"] >= self.confidence_threshold:
                final_predictions.append(original_label)
                continue
                
            # 低置信度触发修正
            if self.chatglm_model:
                correction = self._chatglm_correct(
                    text, 
                    original_label, 
                    bert_result["confidence"]
                )
                
                # 记录修正日志
                log_entry = {
                    "text": text,
                    "original": original_label,
                    "corrected": correction["final_department"],
                    "confidence": bert_result["confidence"],
                    "reason": correction["reason"],
                    "probs": bert_result["probs"].tolist()
                }
                correction_logs.append(log_entry)
                final_predictions.append(correction["final_department"])
            else:
                final_predictions.append(original_label)
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "predictions_with_corrections.json")
        
        result = {
            "predictions": final_predictions,
            "true_labels": test_df['categories'].apply(lambda x: x[0]).tolist(),
            "label_mapping": {i: cls for i, cls in enumerate(self.label_encoder.classes_)},
            "confidence_threshold": self.confidence_threshold,
            "correction_logs": correction_logs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"预测结果已保存至：{output_path}")
        return result

# 使用示例
if __name__ == "__main__":
    # 配置参数

    dataset_name = 'CMDD'


    model_name = 'BERT-chinese'


    bert_model_path = f'个性化分类任务/结果/{model_name}/checkpoint-{dataset_name}'
    chatglm_model_path = "个性化分类任务/模型/glm-4-9b-chat"
    test_file = f'个性化分类任务/数据集/{dataset_name}/test_data_cleaned.json'
    output_dir = "个性化分类任务/结果/BERT+ChatGLM置信度预测"
    
    # 初始化分类器
    classifier = MedicalDepartmentClassifier(
        bert_model_path=bert_model_path,
        chatglm_model_path=chatglm_model_path,
        confidence_threshold=0.75  # 可调节阈值
    )
    
    # 执行预测
    result = classifier.predict_pipeline(test_file, output_dir)



# 执行预测，！！！！！！！！！！！！！！！是否需要添加动态阈值调整
# def dynamic_threshold(text_length):
#     base_threshold = 0.8
#     return base_threshold - min(text_length/500 * 0.1, 0.15)  # 长文本降低阈值，代码如何修改




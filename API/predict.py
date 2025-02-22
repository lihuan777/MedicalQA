from transformers import BertTokenizer, BartForConditionalGeneration
import torch
import os
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import json
import sys



def perdict_text(user_input):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁止输出警告结果
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # 设置GPU从哪张卡能跑
    batch_size = 1  # 设置batch size为1以处理单个用户输入
    accelerator = Accelerator()  # 初始化加速器

    tokenizer = BertTokenizer.from_pretrained("/data/lihuan/MedicalQA/2_model/bart-large-chinese")  # 设置分词器

    # 调整训练模型***************************************需要改********************************************
    generate_model_name = 'model_1Q3A_DPR_20231127_weight_decay_0001'
    
    # 加载模型
    model = BartForConditionalGeneration.from_pretrained(f"/data/lihuan/MedicalQA/2_model/{generate_model_name}/model0")  # 设置模型类别，用于什么任务

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        sys.exit()

    model.to(device)
    model.eval()  # 评估模式

    while True:
        
        
        # 处理用户输入
        inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 生成输出
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("模型输出：", output_text)

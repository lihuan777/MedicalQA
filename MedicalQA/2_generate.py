from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
import torch
import json
import numpy as np
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from evaluation import *
import nltk
from rouge import Rouge
from LoadDataset import LoadDataset
from datetime import date
import sys





def write_file(filename,data):
    with open(filename,'w') as f:
        # f.write('candidata'+'\n')
        for line in data:
            f.write(line+'\n')
            

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  #禁止输出警告结果
    folder = "./"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #设置GPU从哪张卡能跑
    batch_size = 12
    epochs = 1
    accelerator = Accelerator() #初始化加速器
    date = date.today()
    tokenizer = BertTokenizer.from_pretrained("/data/lihuan/MedicalQA/2_model/bart-large-chinese")    #设置分词器

    # 调整训练模型***************************************需要改********************************************
    
    generate_model_name = 'model_1Q3A_DPR_20231127_weight_decay_0001'
    output_name = 'output_1Q3A_DPR_20231127_weight_decay_0001'
    
    # 加载测试数据集
    test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/1_translate/cn_test.json",'r')]
    
    
    model = BartForConditionalGeneration.from_pretrained(f"/data/lihuan/MedicalQA/2_model/{generate_model_name}/model0") #设置模型类别，用于什么任务

    print(model)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        sys.exit()
    print(device)   #输出模型名称
    
    
    
    
    test_dataloader = DataLoader(LoadDataset(test_data, tokenizer,'test'), collate_fn=LoadDataset.collate_fn, batch_size=batch_size,shuffle=False, num_workers=0)

    lr=1e-5 #设置学习率
    optimizer = AdamW(model.parameters(), lr=lr)    #设置优化器

    model, optim,test_dataloader = accelerator.prepare(model, optimizer, test_dataloader)   #将初始数据全部加在到accelerator中    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
    
    for epoch in range(0,epochs):
        accelerator.print(f'train epoch={epoch}')   #输出当前epoch
        losses = []
        if accelerator.is_main_process:
            accelerator.print(f"当前测试第{epoch}轮")
            
            model.eval()    #评估模式

            with torch.no_grad():   #当前计算需要反向传播
                refs = []   #初始化参考答案
                hyps = []   #初始化mask之后的答案
                tk0 = tqdm(test_dataloader, total=len(test_dataloader)) #定义测试集
                

                for batch in tk0:   #对每一个测试集的batch进行测试

                    #获得初值
                    input_ids = batch['input_ids'].long()
                    attention_mask = batch['attention_mask'].long()
                    
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)   #使用模型来生成当前batch的回答
                    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)   #进行解码，是否跳过特殊Token
                    for line in outputs:
                        if line != '':
                            hyps.append(line.replace(' ',''))   #添加一段对话的内容，里面有多个句子
                        else:
                            hyps.append("无")
                
                #先保存decode之后的预测答案***************************************需要改********************************************
                if not os.path.exists(f"/data/lihuan/MedicalQA/3_outputs/{output_name}"): #如果路径不存在输出文件，则新建文件夹
                    os.makedirs(f"/data/lihuan/MedicalQA/3_outputs/{output_name}")
                write_file(f"/data/lihuan/MedicalQA/3_outputs/{output_name}/candidate{epoch}.csv", hyps)   #写入模型文件
                write_file(f"/data/lihuan/MedicalQA/3_outputs/{output_name}/001_candidate{epoch}.txt", hyps)   #写入模型文件



from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
import torch
import json
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


def test(model,dataloader):
    tk0 = tqdm(dataloader, total=len(dataloader))
    res=[]
    for batch in tk0:
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask'].long()
        response = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=512)
        response = tokenizer.batch_decode(response, skip_special_tokens=True)
        for line in response:
            print(line)
            res.append(line)
    return res

def write_file(filename,data):
    with open(filename,'w') as f:
        for line in data:
            f.write(line+'\n')

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  #？？？
    folder = "./"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #设置GPU从哪张卡能跑
    batch_size = 6
    epochs = 1
    model_name = os.path.join(folder,"fnlp-bart-large-chinese/")    #设置模型
    print(model_name)   #打印模型名称
    accelerator = Accelerator() #初始化加速器
    
    tokenizer = BertTokenizer.from_pretrained("/data/lihuan/MedicalQA/2_model/bart-large-chinese")    #设置分词器
    model = BartForConditionalGeneration.from_pretrained("/data/lihuan/MedicalQA/2_model/bart-large-chinese") #设置模型类别，用于什么任务
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU")
        sys.exit()
    print(device)   #输出模型名称
    
    #设置单卡多线程
    # model = nn.DataParallel(model)
    # model = model.cuda()
    # model = model.module.to(device=device)
    
    
    
    #加载数据集***************************************需要改********************************************
    # path ='/data/lihuan/DPR/DPR-main/downloads/data/retriever/cn_test_dpr.json'
    # train_model_name = 'model_ES_2023_06_03_1Q3A'
    # train_data = []
    # with open(path,'r') as file:
    #     str = file.read()
    #     train_data = json.loads(str)
    
    
    # *****************模型名称和训练集的路径都需要改**********************************
    
    
    # 正常
    # train_data = [json.loads(line) for line in open(f"/data/lihuan/DPR/DPR-main/outputs/translate_BART_1Q3A_train.json",'r')]
    
    # 加了医生类型
    train_model_name = 'model_1Q3A_DPR_20231127_weight_decay_0001'
    
    # train_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/1_translate/cn_train_15W_20230918.json",'r')]
    with open('/data/lihuan/DPR/DPR-main/outputs/translate_BART_1Q3A_train_20230922.json', 'r') as json_file:
        train_data = json.load(json_file)
    
    
    
    
    train_dataloder = DataLoader(LoadDataset(train_data,tokenizer,'train'),collate_fn=LoadDataset.collate_fn, batch_size=batch_size,shuffle=False, num_workers=1)

    lr=2e-5 #设置学习率
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0001)    #设置优化器

    model, optim,train_dataloder = accelerator.prepare(model, optimizer, train_dataloder)   #将初始数据全部加在到accelerator中
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloder), num_training_steps=epochs * len(train_dataloder)) #设置线性变化的学习率
    scheduler = accelerator.prepare(scheduler)

    for epoch in range(0,epochs):
        accelerator.wait_for_everyone() #保存模型的必备步骤
        accelerator.print(f'train epoch={epoch}')   #输出当前epoch
        tk0 = tqdm(train_dataloder, total=len(train_dataloder)) #通过tqdm显示加载训练集的进度
        losses = []
        model.train()   #进行训练
        for batch in tk0:
            #得到batch对应的初值
            input_ids = batch['input_ids'].long().to(model.device)
            attention_mask = batch['attention_mask'].long().to(model.device)
            labels = batch['labels'].long().to(model.device)

            # labels[labels == 0] = -100  # 相当于不计算loss

            #输出的赋值
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output.loss 
            loss.requires_grad_(True)   #让 backward 可以追踪这个参数并且计算它的梯度
            # print(loss)
            
            accelerator.backward(loss)  #替代loss.backward()，进行反向传播，计算当前梯度
            optimizer.step()    #更新模型参数
            scheduler.step()    #更新学习率
            optimizer.zero_grad()   #清空之前的梯度
            losses.append(loss.item())  #将当前计算出的loss添加到losses集合中
            tk0.set_postfix(loss=sum(losses)/len(losses))   #在终端显示tqdm的实验进度

        # 保存训练完成的模型
        if not os.path.exists(f"/data/lihuan/MedicalQA/2_model/{train_model_name}"): #如果路径不存在模型文件夹，则新建模型文件夹
            os.makedirs(f"/data/lihuan/MedicalQA/2_model/{train_model_name}")
        model.save_pretrained(f'/data/lihuan/MedicalQA/2_model/{train_model_name}/model{epoch}')  #保存当前Pretrain模型
    print("已完成训练")

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
import numpy as np



def write_file(filename,data):
    with open(filename,'w') as f:
        for line in data:
            f.write(line+'\n')







            
def train(train_dataset,train_model_name):
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
    train_model_name = train_model_name
    
    # train_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/1_translate/cn_train_15W_20230918.json",'r')]
    with open(train_dataset, 'r') as json_file:
        train_data = json.load(json_file)
    
    
    
    
    train_dataloder = DataLoader(LoadDataset(train_data,tokenizer,'train'),collate_fn=LoadDataset.collate_fn, batch_size=batch_size,shuffle=False, num_workers=1)

    lr=2e-5 #设置学习率
    optimizer = AdamW(model.parameters(), lr=lr)    #设置优化器

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









            
def generate(train_model_name,output_name):

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  #禁止输出警告结果
    folder = "./"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #设置GPU从哪张卡能跑
    batch_size = 12
    epochs = 1
    accelerator = Accelerator() #初始化加速器
    tokenizer = BertTokenizer.from_pretrained("/data/lihuan/MedicalQA/2_model/bart-large-chinese")    #设置分词器

    # 调整训练模型***************************************需要改********************************************
    
    train_model_name = train_model_name
    output_name = output_name
    
    # 加载测试数据集
    test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/1_translate/cn_test.json",'r')]
    
    
    model = BartForConditionalGeneration.from_pretrained(f"/data/lihuan/MedicalQA/2_model/{train_model_name}/model0") #设置模型类别，用于什么任务

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


if __name__ == '__main__':

    
    
  
    
    # 高凯老师的任务 
    train_dataset = '/data/lihuan/syncthing/train_dataset.json'
    train_model_name = 'model_BART_20231130'
    output_name = 'output_BART_20231130'
    
    
    train(train_dataset,train_model_name)
    generate(train_model_name,output_name)
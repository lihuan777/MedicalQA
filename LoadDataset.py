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


# 因为答案中存在若干医生的诊断结果，所以需要将所有医生的诊断结果进行合并
def getAnswers(Dataset:list):
    
    answers = ""
    i = len(Dataset)
    
    if i == 0:
        answers = "No Answer"
        return answers
    if i == 1:
        answers = Dataset[0]
        
    if i == 2:
        answers = Dataset[0] + '。' + Dataset[1]
    
    if i >= 3:
        answers = Dataset[0] + '。' + Dataset[1] + '。' + Dataset[2]
        
    return answers
        
    
    
    
    

class LoadDataset(Dataset):
    """
    构造数据集
    """
    def __init__(self,data,tokenizer,type="train"):
        if type=="train":
            self.pair=[{"query":tokenizer.encode(line['ques_title'])
                        ,"answer":tokenizer.encode(getAnswers(line['ans_contents']))
                        } for line in data
                       ]
        else:
            self.pair = [{"query": tokenizer.encode(line['ques_title'])
                             , "answer": tokenizer.encode(getAnswers(line['ans_contents']))
                             } for line in data]
        self.trunck = 512

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, index):
        query=torch.tensor(self.pair[index]['query'][:self.trunck])
        answer=torch.tensor(self.pair[index]['answer'][:self.trunck])
        return query,answer

    @staticmethod
    def collate_fn(data):
        padd_idx=1
        query,answer=zip(*data)
        query=pad_sequence(query,batch_first=True,padding_value=padd_idx)
        answer=pad_sequence(answer,batch_first=True,padding_value=padd_idx)
        return {
            "input_ids": query,
            "attention_mask": query.ne(1),
            "labels": answer
        }



def vocab():
    f=open(r"/data/lihuan/MedicalQA/vocab/wvocab1.txt",encoding='utf-8')
    #创建空列表
    text=[]
    #读取全部内容 ，并以列表方式返回
    lines = f.readlines()      
    for line in lines:
        #如果读到空行，就跳过
        if line.isspace():
            continue
        else:
            #去除文本中的换行等等，可以追加其他操作
            line = line.replace("\n","")
            line = line.replace("\t","")
            #处理完成后的行，追加到列表中
            text.append(line)
    for i in range(len(text)):
        text[i] = text[i].split(" ")[0]
    return text



# 获取医生类型
def getCategories(dataset):
    answers = ""
    for i in range(len(dataset)):
        answer = dataset[i]
        if '科' in answer:
            if i == 0:
                answers = answer
            else:
                answers = answers + " " + answer
            
    return answers
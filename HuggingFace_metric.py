import evaluate
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
# nltk.download('omw-1.4')


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  #？？？
    folder = "./"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #设置GPU从哪张卡能跑
    batch_size = 3
    epochs = 1
    model_name = os.path.join(folder,"fnlp-bart-large-chinese/")    #设置模型
    print(model_name)   #打印模型名称
    accelerator = Accelerator() #初始化加速器

    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-large-chinese")    #设置分词器
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-large-chinese") #设置模型类别，用于什么任务
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #设置优先CUDA
    print(device)   #输出模型名称
    
    #加载数据集
    size = 50000
    test_data = [json.loads(line) for line in open(f"home/kjb/MedicalQA/smallDataset/loc14_test.json",'r')]

    test_dataloader = DataLoader(LoadDataset(test_data, tokenizer,'test'), collate_fn=LoadDataset.collate_fn, batch_size=batch_size,shuffle=False, num_workers=0)

    model,test_dataloader = accelerator.prepare(model, test_dataloader)   #将初始数据全部加在到accelerator中    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
    



    metric = evaluate.load("accuracy")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
import json
import os
import csv
import pandas as pd
import os
# 用loc14数据生成DPR需要的语料库



# test_data = [json.loads(line) for line in open(f"/data/lihuan/DPR/DPR-main/downloads/dataset/cn_train.json",'r',encoding = 'utf-8')]
test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/1_translate/cn_train_20230918.json",'r')]

count = 0


with open('/data/lihuan/DPR/DPR-main/downloads/wikipedia_split/corpus_cn_train.tsv', 'w',newline='',encoding = 'utf-8') as f:
    
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['id', 'text', 'title'])  # 单行写入表头

    for tempData in test_data:
        # 添加最外面一层的数据，因为ans_contents是list，所以分为多条语料库
        count = count + 1
        tsv_w.writerow([str(count), tempData["ans_contents"][0], tempData["ques_title"] ])  # 单行写入

        
        # 进行related字段的语料库添加
        # 20230913 lihuan 发现在related中会包含测试集的问题与答案，所以现在需要在DPR语料库中不包含related相关内容
        # for related in tempData["related"]:
        #     count = count + 1
        #     tsv_w.writerow([str(count), related["ans_contents"][0].replace('\x00', '').replace('\0',''), related["ques_title"].replace('\x00', '').replace('\0','') ])  # 单行写入
        
        
        
        if count%100 == 0:
            print('已处理：' + str(count) )




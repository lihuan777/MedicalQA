import json
import os
import csv
import pandas as pd
import os
import re
# 20230919 lihuan 生成用于检索的QA对
# /data/lihuan/DPR/DPR-main/3_dense_retriever.py



def del_wrong(answer):
    return re.sub(r"[\\']", "", answer)



test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/old/translate/loc14_test.json",'r')]

# with open('/data/lihuan/MedicalQA/1_translate/cn_train_15W_20230918.json', 'r',encoding='utf-8') as file:
#     # 使用json.load()加载JSON数据
#     test_data = json.load(file)





with open('/data/lihuan/DPR/DPR-main/downloads/data/retriever/qas/retriver_test.csv', 'w',newline='',encoding = 'utf-8') as f:
    
    
    tsv_w = csv.writer(f, delimiter='\t')

    for tempData in test_data:
        # 添加最外面一层的数据，因为ans_contents是list，所以分为多条语料库
        answers = ''
        if len(tempData['ans_contents']) > 1:
            
            for i in range(len(tempData['ans_contents'])):
                if i == 0 :
                    answers = "'" + del_wrong(tempData['ans_contents'][i]) + "'"
                else:
                    answers = answers + ",'" + del_wrong(tempData['ans_contents'][i]) + "'"
                
        else:
            answers = "'" + del_wrong(tempData['ans_contents'][0]) + "'"
        
        
        
        answers = '[' + answers + ']'
        
        
        tsv_w.writerow([tempData['ques_title'], answers  ])  # 单行写入

        
        
    print('Done')




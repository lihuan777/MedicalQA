import json
import os
import csv
import sys


    
    
    

path = '/data/lihuan/DPR/DPR-main/downloads/data/retriever/cn_train_dpr.json'
import json
with open(path,'r') as file:
    str = file.read()
    data = json.loads(str)

test_data = [json.loads(line) for line in open(f"/data/lihuan/DPR/DPR-main/outputs/translate_BART_1Q3A_test.json",'r')]


test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/1_translate/fromDPR_BART_1Q3A_train.json",'r')]

print("1")



# path = '/data/lihuan/DPR/DPR-main/downloads/data/retriever/cn_test_dpr.json'

# with open(path,'r',encoding='utf-8') as file:
#     data = json.loads(file.read())

#     with open('/data/lihuan/syncthing_share/cn_test_dpr.json','w',encoding='utf-8') as f:
#         f.write('['+'\n')
#         for i in range(len(data)):
#             data[i]['question'] = data[i]['question'].replace(' ','')
#             for j in range(len(data[i]['answers'])):
#                 data[i]['answers'][j] = data[i]['answers'][j].replace(' ','')
#             f.write(json.dumps(data[i],ensure_ascii=False,indent=4))
#             if i == 4999:
#                 f.write('\n')
#             else:
#                 f.write(','+'\n')

#             if i%100 == 0:
#                 print("Done:" + str(i))
#         f.write(']')
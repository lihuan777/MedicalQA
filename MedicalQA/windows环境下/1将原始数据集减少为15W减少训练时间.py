import sys 
sys.path.append("..")
from LoadDataset import write_file,getAnswers
import json
import os
# 由于原始数据集过大导致训练时间过长，所以设计此功能用于切分原始数据集，有55万，所以用这个代码来保留15万条数据用于实验



# file = open("C:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_train.json", 'r', encoding='utf-8')
# train_data = []
# for line in file.readlines():
#     train_data.append(json.loads(line))


train_data = [json.loads(line) for line in open(f"D:/code/MedicalQA/windows环境下/已整理数据集/2转换为中文的数据集/cn_train.json",'r', encoding='utf-8')]


saveJson = []

for i in range(150000):
    tempJson = {"ques_title":train_data[i]["ques_title"],
             "ques_content":train_data[i]["ques_content"],
             "ans_contents":train_data[i]["ans_contents"],
             "categories":train_data[i]["categories"],
             }
    saveJson.append(tempJson)

    if i%100 == 0:
        print(str(i))
#保存



write_file(f"D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据/cn_train_15W.json", saveJson)   #写入模型文件



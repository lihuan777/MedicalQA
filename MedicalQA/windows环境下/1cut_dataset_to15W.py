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


train_data = [json.loads(line) for line in open(f"C:/code/MedicalQA/已整理数据集/1原始数据集/loc14_train.json",'r', encoding='utf-8')]

count = 0
saveJson = []

for i in range(150000):
    tempJson = {"ques_title":train_data[i]["ques_title"].replace(" ",""),
             "ques_content":train_data[i]["ques_content"].replace(" ",""),
             "ans_contents":getAnswers(train_data[i]["ans_contents"]).replace(" ","")
             }
    saveJson.append(tempJson)
    count = count + 1
    print("进度：" + str(count) + "/150000")
#保存



write_file(f"C:/code/MedicalQA/cn_train_15W.json", saveJson)   #写入模型文件



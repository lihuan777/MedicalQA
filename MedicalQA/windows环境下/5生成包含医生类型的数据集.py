from elasticsearch import Elasticsearch
from LoadDataset import write_file,getAnswers,getCategories
import json
import os
from random import randint
import csv

# 通过医生类型字段的拼接生成新的数据集,需要生成基础的BART训练数据和DPR需要的训练数据


def add_categories(originPath,categoriesPath,savePath):

        
    with open(f"{savePath}",'w',encoding='utf-8') as f:


        # 原始数据
        originData = [json.loads(line) for line in open(originPath,'r',encoding='utf-8')]
        
        # 医生类型的来源
        categoriesData = [json.loads(line) for line in open(categoriesPath,'r',encoding='utf-8')]


        dataSize = 5000
        if len(originData) > 149999:
            dataSize = 150000

        for i in range(dataSize):


            categories = '(' + getCategories(categoriesData[i]['categories']) + ')'
            originData[i]['ques_title'] = categories + originData[i]['ques_title']


            if i%100 == 0:
                print('已处理：' + str(i))


            f.write(json.dumps(originData[i],ensure_ascii=False))
            f.write('\n')
                
            
        print("写入完成！" )


def DPR_add_categories_candidate(originPath,savePath):

    loadData = [json.loads(line) for line in open(originPath,'r',encoding='utf-8')]

    with open(f"{savePath}", "w", encoding="utf-8", newline="") as f:
        
        

        dataSize = 5000
        if len(loadData) > 149999:
            dataSize = 150000

        csv_writer = csv.writer(f)
        for i in range(dataSize):

            line = loadData[i]
            categories = '(' + getCategories(line['categories']) + ')'
            query = categories + line['ques_title'].replace(' ','').replace('"','').replace("'","").replace("＼","").replace('\\','')
            save_data = query + '\t' + '['
            for j in range(len(line['ans_contents'])):
                answer = line['ans_contents'][j].replace(' ','').replace('"','').replace("'","").replace("＼","").replace('\\','')
                if j+1 != len(line['ans_contents']):
                    
                    save_data = save_data + "\'" + answer + "\',"
                else:
                    save_data = save_data + "\'" + answer + "\'"
            
            save_data = save_data + ']'

            csv_writer.writerow([save_data])


            
            if i%100 ==0:
                print(str(i))


if __name__ == '__main__':
    # 处理原始测试集
    originPath = "C:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_test.json"
    categoriesPath = "C:/code/MedicalQA/已整理数据集/1原始数据集/loc14_test.json"
    savePath = "C:/code/MedicalQA/已整理数据集/6添加了医生类型的数据集/cn_test_categories.json"
    add_categories(originPath,categoriesPath,savePath)
    
    # 处理原始训练集15W数据1Q3A
    originPath = "C:/code/MedicalQA/已整理数据集/2dataset_15W/cn_train_15W_3.json"
    categoriesPath = "C:/code/MedicalQA/已整理数据集/1原始数据集/loc14_train.json"
    savePath = "C:/code/MedicalQA/已整理数据集/6添加了医生类型的数据集/cn_train_15W_categories.json"
    add_categories(originPath,categoriesPath,savePath)
    
    # 处理DPR生成的需要进行BART的训练集15W数据1Q3A
    originPath = "/data/lihuan/DPR/DPR-main/outputs/translate_BART_1Q3A_train.json"
    categoriesPath = "C:/code/MedicalQA/已整理数据集/1原始数据集/loc14_train.json"
    savePath = "C:/code/MedicalQA/已整理数据集/6添加了医生类型的数据集/DPR_add_categories_train.json"
    add_categories(originPath,categoriesPath,savePath)


    # 处理DPR生成的需要的数据集candidate_test
    originPath = "C:/code/MedicalQA/已整理数据集/1原始数据集/loc14_test.json"
    savePath = "C:/code/MedicalQA/已整理数据集/6添加了医生类型的数据集/candidate_test_dategories.json"
    # DPR_add_categories_candidate(originPath,savePath)
    # 处理完成后需要手动删除文件内两边的双引号

    # 处理DPR生成的需要的数据集candidate_train
    originPath = "C:/code/MedicalQA/已整理数据集/1原始数据集/loc14_train.json"
    savePath = "C:/code/MedicalQA/已整理数据集/6添加了医生类型的数据集/candidate_train_dategories.json"
    # DPR_add_categories_candidate(originPath,savePath)
    # 处理完成后需要手动删除文件内两边的双引号




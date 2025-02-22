
from elasticsearch import Elasticsearch
import sys 
sys.path.append("..")
from LoadDataset import write_file,getAnswers
import json
import os
from random import randint

# 通过访问ES来查询DPR训练所需要的数据集，其中包含数据"positive_ctxs": [],"negative_ctxs": [],"hard_negative_ctxs": []这些信息并进行拼接

# 定义访问地址
host = "localhost"
port = 9200
elasticSearch = Elasticsearch([{"host":host,"port":port}])



with open(f"D:/code/DPR/DPR-main/downloads/data/retriever/cn_train_onlyDPR.json",'w',encoding='utf-8') as f:

    

    # 定义文件名称
    file_name = "cn_train"

    # 定义索引名称
    index_name = "medical_qa_train"

    

    # loadData = [json.loads(line) for line in open(f"D:/code/lihuan/MedicalQA/1_translate/cn_train_20230918.json",'r',encoding='utf-8')]
    
    with open(f"D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据/cn_train_15W_20230918.json",'r',encoding='utf-8') as trainData:
        loadData = json.load(trainData)

    
    save_json = []


    f.write('['+'\n')

    dataSize = len(loadData)
    for i in range(dataSize):
        line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
        line = json.loads(line)

        line['ques_title'] = loadData[i]['ques_title']
        line['ques_content'] = loadData[i]['ques_content']
        line['ans_contents'] = loadData[i]['ans_contents']
        line['categories'] = loadData[i]['categories']


        # 定义查询结构体
        query = {
            "query": {
                "bool": {
                    "must": [
                        { 
                            "match": { "ques_title": line['ques_title'] } 
                        }
                    ]
                }
            }
        }

        results = elasticSearch.search(index=index_name, body=query, size=30)

        data = {
            "dataset": "medical_qa_train",
            "question": line['ques_title'],
            "answers": loadData[i]['ans_contents'],
            "positive_ctxs": [],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        }

        
        if len(results['hits']['hits']) == 30:
            
            
            temp = {}
            tempData = results['hits']['hits'][0]
            temp = {
                "title":line['ques_title'],
                "text":getAnswers(loadData[i]['ans_contents']),
                "score": 1000,
                "title_score": 1,
                "passage_id": tempData['_id']
                }
            data['positive_ctxs'].append(temp)
        
            
        
        
            for j in range(20):
            #    随机数据作为负样本
                random = randint(0,len(loadData)-1)
                temp = {
                    "title":loadData[random]["ques_title"],
                    "text":getAnswers(loadData[random]['ans_contents']),
                    "score": 0,
                    "title_score": 0,
                    "passage_id": random
                }
                data['negative_ctxs'].append(temp)
                


        
        
            for j in range(20):
                random = randint(0,len(loadData)-1)
                temp = {
                    "title":loadData[random]["ques_title"],
                    "text":getAnswers(loadData[random]['ans_contents']),
                    "score": 0,
                    "title_score": 0,
                    "passage_id": random
                }
                data['hard_negative_ctxs'].append(temp)
        
        # save
        # print(line['ans_contents'])
        save_json.append(data)
        
        if i%100 == 0:
            print('已处理：' + str(i))



        f.write(json.dumps(save_json[i],ensure_ascii=False, indent=4))
            
        if i == dataSize-1:
            f.write('\n')
        else:
            f.write(','+'\n')
    
    f.write(']')
    print("写入完成！" )






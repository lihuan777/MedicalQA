from elasticsearch import Elasticsearch
import json
import os



# 去除重复答案
def clear_repeat(listData):
    tempList = list(set(listData))
    tempList.sort(key = listData.index)
    return tempList



# 通过ES检索数据集中的答案，直接在BART中进行训练
# 此代码为了将训练集的数据进行查询然后拼接为新的数据集之后，生成用于训练的数据

# 定义访问地址
host = "localhost"
port = 9200
elasticSearch = Elasticsearch([{"host":host,"port":port}])




# 定义索引名称
index_name = "cmedqa_train"


# train_data = [json.loads(line) for line in open(f"D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据/cn_train_15W.json",'r', encoding='utf-8')]
with open('D:/code/MedicalQA/cMedQA/datasets/cMedQA_train.json', 'r',encoding='utf-8') as file:
    # 使用json.load()加载JSON数据
    train_data = json.load(file)





saveData = []


for i in range(len(train_data)):

    trainData = train_data[i]
    # 定义查询结构体
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "ques_title":trainData['ques_title']
                        }
                    },
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "ans_contents": trainData['ans_contents'][0]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }

    results = elasticSearch.search(index=index_name, body=query, size=3)


    
    resultData = results['hits']['hits']

    saveJson = trainData

    saveJson['ans_contents'] = []
    
    resultList = []
    for j in range(len(resultData)):
        resultList.extend(resultData[j]['_source']['ans_contents'])
    resultList = clear_repeat(resultList)

    if len(resultList) == 1:
        saveJson['ans_contents'].append(resultList[0])
    
    if len(resultList) == 2:
        saveJson['ans_contents'].append(resultList[0])
        saveJson['ans_contents'].append(resultList[1])

    if len(resultList) > 2:
        saveJson['ans_contents'].append(resultList[0])
        saveJson['ans_contents'].append(resultList[1])
        saveJson['ans_contents'].append(resultList[2])

        
    
        
    saveData.append(saveJson)
    
    if i%1000 == 0:
        print("完成第"+ str(i+1) + "个数据")
    
    





save_path = "D:/code/MedicalQA/cMedQA/datasets/ES检索后直接BART训练.json"
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(saveData, f, ensure_ascii=False, indent=4)

    print("JSON文件已保存至：" + save_path)




print("测试集转换完成")



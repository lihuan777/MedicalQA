from elasticsearch import Elasticsearch
import sys 
sys.path.append("..")
from LoadDataset import write_file
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
index_name = "medical_qa_train"


train_data = [json.loads(line) for line in open(f"D:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_test.json",'r', encoding='utf-8')]
# with open('D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据/cn_train_15W_20230918.json', 'r',encoding='utf-8') as file:
#     # 使用json.load()加载JSON数据
#     train_data = json.load(file)





saveData = []


for i in range(len(train_data)):

    trainData = train_data[i]
    # 定义查询结构体

    # 普通检索
    # query = {
    #     "query": {
    #         "match": {
    #             "ques_title":trainData['ques_title']
    #         }
    #     }
    # }

    # 联合检索
    query = {
        "query": {
            "bool": {
                "must": [
                    { "match": { "ques_title": trainData['ques_title'] } }
                ],
                "filter": [
                    {"terms": {"categories.keyword": trainData['categories'] }}
                    
                ]
            }
        }
    }

    results = elasticSearch.search(index=index_name, body=query, size=30)


# 情况2 Q1-A1,Q1-A2（通过一个Q1索引出相关Q1-Q3，然后用Q1-A1，Q1-A2，Q1-A3，就可以生成3行新数据）
    # ans_contents = ""
    # for i in range(len(results['hits']['hits'])):
    #     data = results['hits']['hits'][i]
    #     ans_contents = ans_contents + data[i]['_source']['ans_contents']
    #     tempData = {
    #         "ques_title":title,
    #         "ans_contents":ans_contents
    #     }
        
    # save_json.append(tempData)
    
    # print("完成第"+ str(i+1) + "个数据，共150000个数据")

# 情况3 Q1-A1+A2+A3（通过一个Q1索引出相关Q1-Q3，然后把三个A拼接为一句话，就生成了一条新的QA）
    
    resultData = results['hits']['hits']

    saveJson = trainData

    saveJson['ans_contents'] = []
    
    resultList = []
    for j in range(len(resultData)):
        resultList.extend(resultData[j]['_source']['ans_contents'])
    resultList = clear_repeat(resultList)


    # 高飞要20条检索数据
    for j in range(len(resultList)):
        saveJson['ans_contents'].append(resultList[j])


    # 这个是拼接三条检索数据
    # if len(resultList) == 1:
    #     saveJson['ans_contents'].append(resultList[0])
    
    # if len(resultList) == 2:
    #     saveJson['ans_contents'].append(resultList[0])
    #     saveJson['ans_contents'].append(resultList[1])

    # if len(resultList) > 2:
    #     saveJson['ans_contents'].append(resultList[0])
    #     saveJson['ans_contents'].append(resultList[1])
    #     saveJson['ans_contents'].append(resultList[2])


    
        
    saveData.append(saveJson)
    
    if i%1000 == 0:
        print("完成第"+ str(i+1) + "个数据，共150000个数据")
    
    


#保存
if not os.path.exists(f"D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据"): #如果路径不存在输出文件，则新建文件夹
    os.makedirs(f"D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据")
write_file(f"D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据/zgf_ES_20240509_cag.json", saveData)   #写入文件
print("测试集转换完成")



from elasticsearch import Elasticsearch
import sys 
sys.path.append("..")
from LoadDataset import write_file,getAnswers
import json
import os


# 通过ES检索数据集中的答案，直接在BART中进行训练
# 此代码为了将训练集的数据进行查询然后拼接为新的数据集之后，生成用于训练的数据

# 定义访问地址
host = "localhost"
port = 9200
elasticSearch = Elasticsearch([{"host":host,"port":port}])




# 定义索引名称
index_name = "medical_qa_train"


train_data = [json.loads(line) for line in open(f"C:/code/MedicalQA/dataset_15W/cn_train_15W.json",'r', encoding='utf-8')]



save_json = []


for i in range(len(train_data)):

    title = train_data[i]['ques_title']
    # 定义查询结构体
    query = {
        "query": {
            "match": {
                "ques_title":title
            }
        }
    }

    results = elasticSearch.search(index=index_name, body=query, size=5)


# 情况2 Q1-A1,Q1-A2（通过一个Q1索引出相关Q1-Q3，然后用Q1-A1，Q1-A2，Q1-A3，就可以生成3行新数据）这种训练方式会训崩
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
    
    tempData = results['hits']['hits']
    tempJson = {
        "ques_title":title,
        "ans_contents":"无答案"
    }
    if len(tempData) == 5:
        tempJson = {
            "ques_title":title,
            "ans_contents":tempData[0]['_source']['ans_contents'] + tempData[1]['_source']['ans_contents'] + tempData[2]['_source']['ans_contents']
        }

        
    
        
    save_json.append(tempJson)
    
    if i%1000 == 0:
        print("完成第"+ str(i+1) + "个数据，共150000个数据")
    
    


#保存
if not os.path.exists(f"C:/code/MedicalQA/dataset_15W"): #如果路径不存在输出文件，则新建文件夹
    os.makedirs(f"C:/code/MedicalQA/dataset_15W")
write_file(f"C:/code/MedicalQA/dataset_15W/cn_train_15W_20230601.json", save_json)   #写入模型文件
print("测试集转换完成")



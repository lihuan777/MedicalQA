from elasticsearch import Elasticsearch
from Function import write_file,getAnswers
import json
import os

# 定义访问地址
host = "localhost"
port = 9200
elasticSearch = Elasticsearch([{"host":host,"port":port}])

# 定义训练数据类型
file = "test"

# 定义文件名称
file_name = "cn_" + file

# 定义索引名称
index_name = "medical_qa_" + file

loadData = [json.loads(line) for line in open(f"/Users/lihuan/Desktop/MedicalQA/dataset/{file_name}.json",'r')]


save_json = []

for i in range(len(loadData)):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = loadData[i]['ques_title'].replace(" ","")
    line['ques_content'] = loadData[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(loadData[i]['ans_contents']).replace(" ","")

    # 定义查询结构体
    query = {
        "query": {
            "match": {
                "ques_title":line['ques_title']
            }
        }
    }

    results = elasticSearch.search(index=index_name, body=query, size=10)


    for data in results['hits']['hits']:
        
        line['ans_contents'] = line['ans_contents'] + "。" + getAnswers(data['_source']['ans_contents']).replace(" ","")
        
    # save
    # print(line['ans_contents'])
    save_json.append(line)


#保存
if not os.path.exists(f"/Users/lihuan/Desktop/MedicalQA/dataset"): #如果路径不存在输出文件，则新建文件夹
    os.makedirs(f"/Users/lihuan/Desktop/MedicalQA/dataset")
write_file(f"/Users/lihuan/Desktop/MedicalQA/dataset/{file_name}_new_dataset.json", save_json)   #写入模型文件
print("测试集转换完成")





from elasticsearch import Elasticsearch
import json
from elasticsearch import helpers

# 创建Elasticsearch客户端
es = Elasticsearch(hosts='http://localhost:9200')

# 打开JSON文件并将其转换为Python对象
loadData = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/old/translate/cn_train_15W.json",'r')]



# 将数据导入Elasticsearch
es.indices.create(index='medical_qa_train')
es.index(index='medical_qa_train', id=1, body={'ques_title': '这是一个标题', 'ques_content': '这是一段内容','ans_contents':'这是回答'})



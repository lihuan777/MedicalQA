import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers

es = Elasticsearch(hosts='http://localhost:9200')

# 创建索引dpr_train
# res = es.indices.create(index="medical_qa_train")
# print(res)

# 查询索引
# 使用search查询数据
query = {
    "query": {
        "match_all": {}
    }
}
res = es.search(index="medical_qa_train", body=query)
print(res['hits']['hits'])
import json
# 在终端中执行下列操作

curl --location 'http://localhost:9200/medical_qa_train/_bulk' \
--header 'Content-Type: application/json' \
--data '@/data/lihuan/MedicalQA/old/translate/cn_train_15W_ES1.json'

curl -H "Content-Type: application/json" -XPOST 'http://localhost:9200/medical_qa_train/_bulk?pretty' --data-binary '@/data/lihuan/MedicalQA/old/translate/loc14_train_QA0.json'
curl -u elastic -H "Content-Type: application/json"  -XPOST "IP:9200/medical_qa_train/_bulk" --data-binary @/data/lihuan/MedicalQA/old/translate/cn_train_15W_ES1.json

curl --location 'http://localhost:9200/medical_qa_train/_bulk' \
--header 'Content-Type: application/json' \
--data '@/C:/code/MedicalQA/dataset_15W/cn_train_15W_ES1.json'


# 查看所有索引
curl -XGET 'http://localhost:9200/_cat/indices?v'

# 创建索引
curl XPUT 'http://localhost:9200/medical_qa_train'

# 删除索引
curl -XDELETE 'http://localhost:9200/medical_qa_train'

# 查询指定索引的内容
curl --location 'http://localhost:9200/medical_qa_train/_search'


# 查看指定索引中的数据条数
curl -X GET "http://localhost:9200/medical_qa_train/_count" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match_all": {}
  }
}
'





from elasticsearch import Elasticsearch
import json
import os
from random import randint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 用于显示进度条

# 通过访问ES来查询DPR训练所需要的数据集，其中包含数据"positive_ctxs": [],"negative_ctxs": [],"hard_negative_ctxs": []这些信息并进行拼接

# 定义访问地址
host = "localhost"
port = 9200
elasticSearch = Elasticsearch([{"host": host, "port": port}])

# 定义索引名称
index_name = "cmedqa_train"

with open(f"cMedQA/datasets/cMedQA_train.json", 'r', encoding='utf-8') as trainData:
    loadData = json.load(trainData)

# 定义保存路径
output_path = f"D:/code/DPR/DPR-main/downloads/data/retriever/cMedQA_train_dpr.json"

# 查询并处理每条数据的函数
def process_data(i, loadData):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":[]}'
    line = json.loads(line)

    line['ques_title'] = loadData[i]['ques_title']
    line['ans_contents'] = loadData[i]['ans_contents']
    line['categories'] = loadData[i]['categories']
    line['ans_descriptions'] = loadData[i]['ans_descriptions']

    # 定义查询结构体
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"ques_title": loadData[i]['ques_title']}},
                    {
                        "bool": {
                            "should": [{"match": {"ans_contents": loadData[i]['ans_contents'][0]}}]
                        }
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
        for j in range(10):
            temp = {}
            if j == 0:
                tempData = results['hits']['hits'][j]
                temp = {
                    "title": line['ques_title'],
                    "text": loadData[i]['ans_contents'][0],
                    "score": 1000,
                    "title_score": 1,
                    "passage_id": tempData['_id']
                }
                data['positive_ctxs'].append(temp)
            else:
                tempData = results['hits']['hits'][j]
                temp = {
                    "title": tempData['_source']["ques_title"],
                    "text": tempData['_source']['ans_contents'][0],
                    "score": tempData['_score'],
                    "title_score": 0,
                    "passage_id": tempData['_id']
                }
                data['positive_ctxs'].append(temp)

        for j in range(20):
            # 随机数据作为负样本
            random = randint(0, len(loadData) - 1)
            temp = {
                "title": loadData[random]["ques_title"],
                "text": loadData[random]['ans_contents'][0],
                "score": 0,
                "title_score": 0,
                "passage_id": random
            }
            data['negative_ctxs'].append(temp)

        for j in range(20):
            # 取30条数据的后20条作为困难的负样本
            tempData = results['hits']['hits'][j + 10]
            temp = {
                "title": tempData['_source']["ques_title"],
                "text": tempData['_source']['ans_contents'][0],
                "score": tempData['_score'],
                "title_score": 0,
                "passage_id": tempData['_id']
            }
            data['hard_negative_ctxs'].append(temp)

    return i, data  # 返回数据的索引和处理后的数据


# 多线程执行数据处理
def main():
    save_json = [None] * len(loadData)  # 初始化一个列表以便按照索引存储数据
    dataSize = len(loadData)

    with ThreadPoolExecutor(max_workers=5) as executor:  # 使用5个线程
        futures = {executor.submit(process_data, i, loadData): i for i in range(dataSize)}

        with tqdm(total=dataSize, desc="Processing") as pbar:
            for future in as_completed(futures):
                i, data = future.result()
                save_json[i] = data  # 根据索引存储数据
                pbar.update(1)

    # 写入结果
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, data in enumerate(save_json):
            f.write(json.dumps(data, ensure_ascii=False, indent=4))
            if i < dataSize - 1:
                f.write(',\n')
        f.write('\n]')
        print("写入完成！")


if __name__ == "__main__":
    main()

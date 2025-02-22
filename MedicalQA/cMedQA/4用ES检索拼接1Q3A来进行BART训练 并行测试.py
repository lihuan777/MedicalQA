from elasticsearch import Elasticsearch
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 进度条库

# 去除重复答案
def clear_repeat(listData):
    tempList = list(set(listData))
    tempList.sort(key = listData.index)
    return tempList

# 定义访问地址
host = "localhost"
port = 9200
elasticSearch = Elasticsearch([{"host": host, "port": port}])

# 定义索引名称
index_name = "cmedqa_train"

# 读取训练数据
with open('D:/code/MedicalQA/cMedQA/datasets/cMedQA_train.json', 'r', encoding='utf-8') as file:
    train_data = json.load(file)

saveData = [None] * len(train_data)  # 使用预分配的列表来保存结果，保证顺序

# 定义函数，用于并行处理每一条数据
def process_train_data(i):
    trainData = train_data[i]
    # 定义查询结构体
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "ques_title": trainData['ques_title']
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

    # 执行查询
    results = elasticSearch.search(index=index_name, body=query, size=3)
    resultData = results['hits']['hits']

    saveJson = trainData.copy()
    saveJson['ans_contents'] = []

    resultList = []
    for result in resultData:
        resultList.extend(result['_source']['ans_contents'])
    resultList = clear_repeat(resultList)

    # 保存前3条结果
    saveJson['ans_contents'] = resultList[:3]

    return i, saveJson  # 返回索引和结果

# 初始化进度条
total_data = len(train_data)

# 使用线程池并行处理，并输出进度
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_train_data, i) for i in range(total_data)]

    with tqdm(total=total_data, desc="Processing data") as pbar:
        for future in as_completed(futures):
            try:
                index, result = future.result()
                saveData[index] = result  # 根据索引保存结果，确保顺序一致
            except Exception as exc:
                print(f"处理数据时出现异常: {exc}")
            # 每完成一个任务，更新进度条
            pbar.update(1)

# 保存结果
save_path = "D:/code/MedicalQA/cMedQA/datasets/ES检索后直接BART训练.json"
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(saveData, f, ensure_ascii=False, indent=4)
    print("JSON文件已保存至：" + save_path)

print("测试集转换完成")

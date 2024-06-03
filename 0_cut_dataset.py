# 将数据集处理拼接答案。例如Q-3A
from elasticsearch import Elasticsearch
from LoadDataset import getAnswers
import json
import os



# 此代码为了将训练集的数据进行查询然后拼接为新的数据集之后，生成用于训练的数据



train_data = [json.loads(line) for line in open(f"/data/lihuan/DPR/DPR-main/outputs/dense_retriver_test.json",'r', encoding='utf-8')]



save_json = []


# 情况1，直接使用QA进行训练
for i in range(len(train_data)):
    data = train_data[i]
    ans_contents = ans_contents + data[i]['_source']['ans_contents']
    tempData = {
        "ques_title":title,
        "ans_contents":ans_contents
    }


# 情况2 Q1-A1,Q1-A2（通过一个Q1索引出相关Q1-Q3，然后用Q1-A1，Q1-A2，Q1-A3，就可以生成3行新数据）
# ans_contents = ""
# for i in range(len(train_data)):
#     data = results['hits']['hits'][i]
#     ans_contents = ans_contents + data[i]['_source']['ans_contents']
#     tempData = {
#         "ques_title":title,
#         "ans_contents":ans_contents
#     }
    
# save_json.append(tempData)

print("完成第"+ str(i+1) + "个数据，共150000个数据")

# 情况3 Q1-A1+A2+A3（通过一个Q1索引出相关Q1-Q3，然后把三个A拼接为一句话，就生成了一条新的QA）
# for data in results['hits']['hits']:
#     tempData = {
#         "ques_title":title,
#         "ans_contents":data['_source']['ans_contents']
#     }
    
#     save_json.append(tempData)

# print("完成第"+ str(i+1) + "个数据，共150000个数据")
    


#保存
if not os.path.exists(f"C:/code/MedicalQA/dataset_15W"): #如果路径不存在输出文件，则新建文件夹
    os.makedirs(f"C:/code/MedicalQA/dataset_15W")
write_file(f"C:/code/MedicalQA/dataset_15W/cn_train_15W_3.json", save_json)   #写入模型文件
print("测试集转换完成")



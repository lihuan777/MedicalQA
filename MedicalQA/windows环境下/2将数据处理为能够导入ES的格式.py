import sys 
sys.path.append("..")
from LoadDataset import write_ES,getAnswers,getCategories
import json
import os


# 此代码为了将中文数据转换为用Postman能够导入ElasticSearch的格式

# def write_file(filename,data,count):

#     # 添加计数器，防止转换之后出现多余空行导致报错
#     sum = len(data)

#     with open(filename,'w') as f:
#         for line in data:
            
#             index = '{ "index" : { "_index" : "medical_qa_train", "_id": ' + str(count) + '} }'
#             # index = json.loads(index)
#             f.write(index+'\n')
#             f.write(str(line)+'\n')
#             count = count + 1
#             # if count != sum:
#             #     f.write(json.dumps(line,ensure_ascii=False)+'\n')
#             # else:
#             #     f.write(json.dumps(line,ensure_ascii=False))
#         print("转换完成！已转换" + str(count) + "条数据")



def cut(save_file,origin_file,num):

    train_data = [json.loads(line) for line in open(f"{origin_file}",'r', encoding='utf-8')]

    start = 0
    end = 99999

    
    if num == 1:
        start = 100000
        end = 199999
    if num == 2:
        start = 200000
        end = 299999
    if num == 3:
        start = 300000
        end = 399999
    if num == 4:
        start = 400000
        end = 499999
    if num == 5:
        start = 500000
        end = len(train_data)-1

    
    save_data = []

    with open(f"{save_file}",'w',encoding='utf-8',newline='\n') as f:
        for i in range(start,end+1):
            tempJson = {
                "ques_title":train_data[i]["ques_title"],
                "ques_content":train_data[i]["ques_content"],
                "ans_contents":train_data[i]["ans_contents"],
                "categories":train_data[i]['categories']
            }
    


            index = '{"index" : { "_index" : "medical_qa_train", "_id": ' + str(i) + '} }'
            
            if i%100 == 0:
                print("Rencent：" + str(i))

            f.write(index + '\n')
            f.write(json.dumps(tempJson,ensure_ascii=False) + '\n')
        
            if i%100 == 0:
                print("Done:" + str(i))






for i in range(6):
    origin_file = "D:/code/MedicalQA/已整理数据集/去掉训练集中测试集数据/cn_train.json"
    save_path = "D:/code/MedicalQA/已整理数据集/4中文数据集只保留QA对的数据集，用于Postman导入ES"
    if not os.path.exists(f"{save_path}"): #如果路径不存在输出文件，则新建文件夹
        os.makedirs(f"{save_path}")

    save_file = save_path + "/cn_train_QA"+ str(i) + "_import.json"

    cut(save_file,origin_file,i)






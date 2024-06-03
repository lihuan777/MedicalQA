import json
import os
from LoadDataset import LoadDataset
from LoadDataset import getAnswers
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

def write_file(filename,data,count):

    # 添加计数器，防止转换之后出现多余空行导致报错
    sum = len(data)

    with open(filename,'w') as f:
        for line in data:
            
            index = '{ "index" : { "_index" : "medical_qa_train", "_id": ' + str(count) + '} }'
            index = json.loads(index)
            f.write(json.dumps(index,ensure_ascii=False)+'\n')
            f.write(json.dumps(line,ensure_ascii=False)+'\n')
            count = count + 1
            # if count != sum:
            #     f.write(json.dumps(line,ensure_ascii=False)+'\n')
            # else:
            #     f.write(json.dumps(line,ensure_ascii=False))
        print(path_name + "转换完成！已转换" + str(count) + "条数据")


# 
file  = "train"

path_name = "cn_" + file

test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/translate/{path_name}.json",'r')]



save_json1 = []
save_json2 = []
save_json3 = []
save_json4 = []
save_json5 = []
save_json6 = []

# for i in range(len(test_data)):
for i in range(0,100000):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = test_data[i]['ques_title'].replace(" ","")
    line['ques_content'] = test_data[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(test_data[i]['ans_contents']).replace(" ","")

    save_json1.append(line)

for i in range(100000,200000):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = test_data[i]['ques_title'].replace(" ","")
    line['ques_content'] = test_data[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(test_data[i]['ans_contents']).replace(" ","")

    save_json2.append(line)

for i in range(200000,300000):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = test_data[i]['ques_title'].replace(" ","")
    line['ques_content'] = test_data[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(test_data[i]['ans_contents']).replace(" ","")

    save_json3.append(line)

for i in range(300000,400000):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = test_data[i]['ques_title'].replace(" ","")
    line['ques_content'] = test_data[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(test_data[i]['ans_contents']).replace(" ","")

    save_json4.append(line)

for i in range(400000,500000):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = test_data[i]['ques_title'].replace(" ","")
    line['ques_content'] = test_data[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(test_data[i]['ans_contents']).replace(" ","")

    save_json5.append(line)

for i in range(500000,len(test_data)):
    line = '{"ques_title":"0","ques_content":"0","ans_contents":"0"}'
    line = json.loads(line)
    line['ques_title'] = test_data[i]['ques_title'].replace(" ","")
    line['ques_content'] = test_data[i]['ques_content'].replace(" ","")
    line['ans_contents'] = getAnswers(test_data[i]['ans_contents']).replace(" ","")

    save_json6.append(line)

# print(save_json)

# batch_size = 3
# tokenizer = BertTokenizer.from_pretrained("fnlp/bart-large-chinese")    # 设置分词器

# train_dataloder = DataLoader(LoadDataset(test_data,tokenizer,'train'),collate_fn=LoadDataset.collate_fn, batch_size=batch_size,shuffle=False, num_workers=1)



#保存
if not os.path.exists(f"/data/lihuan/MedicalQA/translate"): #如果路径不存在输出文件，则新建文件夹
    os.makedirs(f"/data/lihuan/MedicalQA/translate")
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}_QA1.json", save_json1,1)   #写入模型文件
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}_QA2.json", save_json2,100001)   #写入模型文件
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}_QA3.json", save_json3,200001)   #写入模型文件
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}_QA4.json", save_json4,300001)   #写入模型文件
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}_QA5.json", save_json5,400001)   #写入模型文件
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}_QA6.json", save_json6,500001)   #写入模型文件

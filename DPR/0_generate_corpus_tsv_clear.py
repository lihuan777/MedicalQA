# 这个文件是用来生成清洗去重的语料库的，因为不知道语料库是否有重复会不会对生成效果有影响
import json
import csv


# 获取医生类型
def getCategories(dataset):
    answers = ""
    for i in range(len(dataset)):
        answer = dataset[i]
        if '科' in answer:
            if i == 0:
                answers = answer
            else:
                answers = answers + " " + answer
            
    return answers




train_data = [json.loads(line) for line in open(f"/data/home/zgf/qa/bart-chinese/smallDataset/loc14_train.json",'r')]
val_data = [json.loads(line) for line in open(f"/data/home/zgf/qa/bart-chinese/smallDataset/loc14_val.json",'r')]
qas=[]
for i in range(len(train_data)):
    re=train_data[i]['related']
    for k in range(len(re)):
        categories = '(' + getCategories(re[k]['categories']) + ')'
        
        q=categories + re[k]['ques_title']
        a=""
        for j in range(len(re[k]['ans_contents'])):
            a+=re[k]['ans_contents'][j]
        if(q=="" or a==""):
            continue
        qas.append(q.replace(' ','')+"/t"+a.replace(' ',''))
for i in range(len(val_data)):
    re=val_data[i]['related']
    for k in range(len(re)):
        categories = '(' + getCategories(re[k]['categories']) + ')'
        q=categories + re[k]['ques_title']
        a=""
        for j in range(len(re[k]['ans_contents'])):
            a+=re[k]['ans_contents'][j]
        if(q=="" or a==""):
            continue
        qas.append(q.replace(' ','')+"/t"+a.replace(' ',''))
qa=set(qas)
qa=list(qa)



count = 0

with open(f'/data/lihuan/DPR/DPR-main/downloads/wikipedia_split/corpus_cn_train_clear_categories.tsv', 'w',newline='',encoding = 'utf-8') as f:
    
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow(['id', 'text', 'title'])  # 单行写入表头

    for tempData in qa:
        # 添加最外面一层的数据，因为ans_contents是list，所以分为多条语料库
        count = count + 1
        tsv_w.writerow([str(count), tempData.split("/t")[1], tempData.split("/t")[0]])  # 单行写入

        print('已处理：' + str(count) )



print("Task Done")
import json

def write_file(file_name,data):


    with open(file_name,'w',encoding='utf-8') as f:
        # for line in data:
        #     f.write(json.dumps(line,ensure_ascii=False)+'\n')
        f.write(json.dumps(data,ensure_ascii=False, indent=4))
        print(file_name + "Done" )



# 因为答案中存在若干医生的诊断结果，所以需要将所有医生的诊断结果进行合并
def getAnswers(Dataset):
    answers = ""
    for answer in Dataset:
        answers = answers  + answer + "。"
    return answers


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



def write_ES(file_name,data):


    with open(file_name,'w',encoding='utf-8') as f:
        for i in data:

            f.write(json.dumps(data[i],ensure_ascii=False))
        print(file_name + "写入完成！" )
import json
from LoadDataset import LoadDataset



def get3Answers(Dataset:list):
    
    answers = ""
    i = len(Dataset)
    
    if i == 0:
        answers = "No Answer"
        return answers
    if i == 1:
        answers = Dataset[0]
        
    if i == 2:
        answers = Dataset[0] + '。' + Dataset[1]
    
    if i >= 3:
        answers = Dataset[0] + '。' + Dataset[1] + '。' + Dataset[2]
        

    return answers # 取3条答案


# 读取数据

with open("D:/code/MedicalQA/cMedQA/datasets/cMedQA_test(标准json).json", 'r', encoding='utf-8') as file:


    train_data = json.load(file)


    for i in range(len(train_data)):
        # 假设每个数据项中都有一个'ans_contents'字段

        train_data[i]['ans_contents'] = get3Answers(train_data[i]['ans_contents'])

        


# 保存到新的JSON文件
with open('D:/code/MedicalQA/cMedQA/datasets/cMedQA_test.json', 'w', encoding='utf-8') as output_file:
    for line in train_data:
        if isinstance(line, dict):
            line = json.dumps(line, ensure_ascii=False)
        else:
            line = str(line)
        output_file.write(line + '\n')





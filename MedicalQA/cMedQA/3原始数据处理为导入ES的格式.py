import sys 
import json
import os





def cut(save_file,origin_file):


    with open(origin_file, 'r', encoding='utf-8') as file:
        train_data = json.load(file)

    

    with open(save_file,'w',encoding='utf-8',newline='\n') as f:
        for i in range(len(train_data)):
            tempJson = {
                "ques_title":train_data[i]["ques_title"],
                "ans_contents":train_data[i]["ans_contents"],
                "categories":[],
                "ans_descriptions":[]
            }
    


            index = '{"index" : { "_index" : "cmedqa_train", "_id": ' + str(i) + '} }'
            
            if i%100 == 0:
                print("Rencent：" + str(i))

            f.write(index + '\n')
            f.write(json.dumps(tempJson,ensure_ascii=False) + '\n')
        
            if i%100 == 0:
                print("Done:" + str(i))







origin_file = "cMedQA/datasets/cMedQA_train.json"
save_path = "cMedQA/datasets/postman导入ES的数据"

save_file = save_path + "/cMedQA_train_" + "importES.json"

cut(save_file,origin_file)






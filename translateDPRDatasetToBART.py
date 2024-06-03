import json


# 将DPR输入的数据集生成BART的数据集
# ******************************************需要修改*********************************************





def translate(file,save_file):
    f1 = open(file,'r')
    json_data = json.load(f1)
    save_json = []


    # 这是1Q3A的情况
    with open(save_file,"w",encoding='utf-8') as f:
        for i in range(len(json_data)):
            if len(json_data[i]['positive_ctxs']) >= 2:
                
                save_json.append({
                    "ques_title": json_data[i]["question"]  ,
                    "ans_contents":json_data[i]['answers'][0] + '。' + json_data[i]['positive_ctxs'][1]["text"] + '。' + json_data[i]['positive_ctxs'][2]["text"]
                })
            else:
                save_json.append({
                    "ques_title": json_data[i]["question"]  ,
                    "ans_contents":json_data[i]['answers'][0]
                })
                print("在" + str(i) + "处存在无positive_ctxs")
            
            f.write(json.dumps(save_json[i],ensure_ascii=False)+'\n')
            if i%100 == 0:
                print('已处理train：' + str(i))
        
        print("已处理完成，路径为" + save_file)

if __name__ == "__main__":
    
    file = '/data/lihuan/DPR/DPR-main/downloads/data/retriever/cn_train_dpr.json'
    save_file = '/data/lihuan/MedicalQA/1_translate/ES_BART_1Q3A_train.json'
    translate(file,save_file)
    
import json
import sys


# lihuan用于在DPR检索的数据之后进行医生类型的拼接然后进行BART训练
# ******************************************需要修改*********************************************


def getCategories(Dataset):
    categories = ""
    for category in Dataset:
        categories = categories + " " + category
    return '[' + categories + ']'


def translate(origin_file,categories_file,save_file):
    origin_file = open(origin_file,'r')
    origin_data = json.load(origin_file)
    
    
    categories_file = open(categories_file,'r')
    categories_data = json.load(categories_file)
    
    save_json = []

    # 这是1Q3A的情况
    with open(save_file,"w",encoding='utf-8') as f:
        
        
        for i in range(len(origin_data)):
            
            # answer_num = len()
            origin_data[i]['ques_title'] =  origin_data[i]['ques_title'] + getCategories(categories_data[i]['categories'])
            
            if i % 1000 == 0:
                print("Done:" + str(i))
       
        f.write(json.dumps(origin_data,ensure_ascii=False, indent=4))
        
        
        

if __name__ == "__main__":
    
    
    origin_file = '/data/lihuan/DPR/DPR-main/outputs/translate_BART_1Q3A_categories_20231016.json'
    categories_file = '/data/lihuan/MedicalQA/1_translate/cn_train_15W_20230918.json'
    save_file = '/data/lihuan/DPR/DPR-main/outputs/translate_BART_1Q3A_categories_rear_20231024.json'
    translate(origin_file,categories_file,save_file)
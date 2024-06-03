import json
import sys


# 20231028lihuan用于生成张高飞需要大模型训练的关联医生类型的数据
# ******************************************需要修改*********************************************

# 获取医生类型字段并拼接
def getCategories(Dataset):
    categories = ""
    for category in Dataset:
        categories = categories + " " + category
    return '[' + categories + ']'

# 前置拼接
def combine_front(file,categories_file,save_file):
    f1 = open(file,'r',encoding='utf-8')
    json_data = json.load(f1)
    save_json = []
    
    
    f2 = open(categories_file,'r',encoding='utf-8')
    categoriesn_data = json.load(f2)
  

    # 进行拼接情况的处理
    with open(save_file,"w",encoding='utf-8') as f:
        
        
        for i in range(len(json_data)):
            
            save_json.append({
                "ques_title":  getCategories(categoriesn_data[i]['categories']) + json_data[i]["ques_title"],
                "ans_contents":json_data[i]['ans_contents']
            })
            
            

        f.write(json.dumps(save_json,ensure_ascii=False, indent=4))
        
        print("已处理完成" + save_file)



# 后置拼接
def combine_rear(file,categories_file,save_file):
    f1 = open(file,'r',encoding='utf-8')
    json_data = json.load(f1)
    save_json = []
    
    
    f2 = open(categories_file,'r',encoding='utf-8')
    categoriesn_data = json.load(f2)
  

    # 进行拼接情况的处理
    with open(save_file,"w",encoding='utf-8') as f:
        
        
        for i in range(len(json_data)):
            
            save_json.append({
                "ques_title":  json_data[i]["ques_title"] + getCategories(categoriesn_data[i]['categories']),
                "ans_contents":json_data[i]['ans_contents']
            })
            
            

        f.write(json.dumps(save_json,ensure_ascii=False, indent=4))
        
        print("已处理完成" + save_file)



if __name__ == "__main__":
    
   
    
    file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_20231027.json'
    categories_file = 'D:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_test_sample.json'
    save_file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_categories_front_20231028.json'
    combine_front(file,categories_file,save_file)
    
    file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_20231027.json'
    categories_file = 'D:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_test_sample.json'
    save_file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_categories_rear_20231028.json'
    combine_rear(file,categories_file,save_file)
    
    file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_categories_20231026.json'
    categories_file = 'D:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_test_sample.json'
    save_file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_es_categories_front_20231028.json'
    combine_front(file,categories_file,save_file)
    
    file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_categories_20231026.json'
    categories_file = 'D:/code/MedicalQA/已整理数据集/2转换为中文的数据集/cn_test_sample.json'
    save_file = 'D:/code/lihuan/DPR/DPR-main/outputs/zgf_test_es_categories_rear_20231028.json'
    combine_rear(file,categories_file,save_file)
    
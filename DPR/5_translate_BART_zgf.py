import json
import sys



# ******************************************需要修改*********************************************





def translate(file,save_file):
    f1 = open(file,'r')
    json_data = json.load(f1)
    save_json = []
    
    
    
  


    # 这是1Q3A的情况
    with open(save_file,"w",encoding='utf-8') as f:
        
        
        for i in range(len(json_data)):
            
            # answer_num = len()
            
            tempList = []
            for temp in json_data[i]['ctxs']:
                tempList.append(temp['text'])
            
            answerList = list(set(tempList))
            answerList.sort(key = tempList.index)
            
            
            
            answers = answerList[:3]
            
            
            
            
            
            
            
            save_json.append({
                "ques_title": json_data[i]["question"],
                "ans_contents":answers
            })
            
            

        f.write(json.dumps(save_json,ensure_ascii=False, indent=4))
        
        print("已处理完成" + save_file)

if __name__ == "__main__":
    
   
    
    file = '/data/lihuan/DPR/DPR-main/outputs/zgf_test.json'
    save_file = '/data/lihuan/DPR/DPR-main/outputs/zgf_test_20231027.json'
    translate(file,save_file)
import json
import sys

# 此文件是为了训练Reader Model时排除在dense retriver生成的数据集中存在空白的情况


path='/data/lihuan/DPR/DPR-main/outputs/dense_retriver_train.json' 
f1=open(path,'r')
data=json.load(f1)



    
    
file_name = "/data/lihuan/DPR/DPR-main/outputs/dense_retriver_train_NotNone.json"
with open(file_name,"w",encoding='utf-8') as f:
    
    
    
    for i in range(len(data)):
        
        
        
        if len(data[i]['answers']) == 0 or len(data[i]['question']) == 0 or len(data[i]['ctxs']) != 10:
            # 上述元素存在空值，退出程序
            print(str(i) + "为空")
            sys.exit()
        else:
            # 上述元素均不为空，继续判断ctxs中的内容
            for j in range(10):
                if len(data[i]['ctxs'][j]['title']) == 0:
                    data[i]['ctxs'][j]['title'] = data[i]['question']
                    print("已处理的数据为第" + str(i) + "个数据的第" + str(j) + "个ctxs的title" )

                
                if len(data[i]['ctxs'][j]['text']) == 0:
                    data[i]['ctxs'][j]['text'] = data[i]['answers']
                    print("已处理的数据为第" + str(i) + "个数据的第" + str(j) + "个ctxs的text" )
        
    
    
    # f.write(json.dumps(data,ensure_ascii=False,indent=4))
    f.write(json.dumps(data,indent=4))
    
print("Mission Done")
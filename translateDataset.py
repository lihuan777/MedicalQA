import json
import os



path_name = "loc14_val.json"

test_data = [json.loads(line) for line in open(f"/data/Dataset/MedicalQA/{path_name}",'r')]



def write_file(filename,data):

    # 添加计数器，防止转换之后出现多余空行导致报错
    count = 0
    sum = len(data)

    with open(filename,'w') as f:
        for line in data:
            count = count + 1
            if count != sum:
                f.write(json.dumps(line,ensure_ascii=False)+'\n')
            else:
                f.write(json.dumps(line,ensure_ascii=False))
        print(path_name + "转换完成！已转换" + str(count) + "条数据")

#保存
if not os.path.exists(f"/data/lihuan/MedicalQA/translate"): #如果路径不存在输出文件，则新建文件夹
    os.makedirs(f"/data/lihuan/MedicalQA/translate")
write_file(f"/data/lihuan/MedicalQA/translate/{path_name}", test_data)   #写入模型文件





from pyrouge import Rouge155
import os





# 同时读取文件‘二次检索任务/1 原始数据.json’作为生成文件，其中的ans_contents中存在3条数据，先将所有数据的每一条数据的ans_contents的第一个元素取出来共计与测试集进行评估，再去除每个数据的ans_contents的第二条进行评估，再取出每个数据的ans_contents的第三条进行评估。得到三组rouge结果后对每一个rouge参数进行平均。


    
r = Rouge155()


r.rouge_dir = f'D:/code/ROUGE-RELEASE-1.5.5'  # 添加这行代码


# 测试数据集
r.model_dir = '二次检索任务/数据集/reference'


r.model_filename_pattern = 'reference_3000.txt'


# 生成数据集
r.system_dir = '二次检索任务/数据集/2'


r.system_filename_pattern = '(\d+)_candidate0.txt'



output = r.convert_and_evaluate()

print(output)

output_dict = r.output_to_dict(output)


# 提取rouge_1, rouge_2, rouge_l
rouge_1 = {
    'rouge_1_recall': round(output_dict['rouge_1_recall'] * 100, 2),
    'rouge_1_f_score': round(output_dict['rouge_1_f_score'] * 100, 2)
}

rouge_2 = {
    'rouge_2_recall': round(output_dict['rouge_2_recall'] * 100, 2),
    'rouge_2_f_score': round(output_dict['rouge_2_f_score'] * 100, 2)
}

rouge_l = {
    'rouge_l_recall': round(output_dict['rouge_l_recall'] * 100, 2),
    'rouge_l_f_score': round(output_dict['rouge_l_f_score'] * 100, 2)
}

# 输出结果
print(rouge_1)
print(rouge_2)
print(rouge_l)




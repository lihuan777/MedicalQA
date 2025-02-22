from pyrouge import Rouge155
import os
import re


    
r = Rouge155()

# 设置 ROUGE 的路径
r.rouge_dir = 'D:/code/ROUGE-RELEASE-1.5.5'  # 添加这行代码

# 测试数据集

r.model_dir = '/data/lihuan/ROUGE/reference'

r.model_filename_pattern = '001_reference.txt'


r.system_dir = '/data/lihuan/MedicalQA/3_outputs/output_BART_20231130'

# 正确的正则表达式格式
r.system_filename_pattern = '(\d+)_candidate0.txt'

output = r.convert_and_evaluate()

print(output)

output_dict = r.output_to_dict(output)
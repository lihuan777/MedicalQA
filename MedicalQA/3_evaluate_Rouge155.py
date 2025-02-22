from pyrouge import Rouge155
import os
import re

def get_os_type():
    if os.name == 'nt':
        return 'Windows'
    elif os.name == 'posix':
        return 'Linux'
    
r = Rouge155()

# 设置 ROUGE 的路径
r.rouge_dir = 'D:/code/ROUGE-RELEASE-1.5.5'  # 添加这行代码

# 测试数据集
os_type = get_os_type()  # 正确调用函数
if os_type == 'Windows':
    r.model_dir = 'D:/code/ROUGE/reference'
elif os_type == 'Linux':
    r.model_dir = '/data/lihuan/ROUGE/reference'

r.model_filename_pattern = '001_reference.txt'

# 生成数据集
if os_type == 'Windows':
    r.system_dir = 'D:/code/MedicalQA/cMedQA/3_outputs/output_1Q1A_2024102501'
elif os_type == 'Linux':
    r.system_dir = '/data/lihuan/MedicalQA/3_outputs/output_BART_20231130'

# 正确的正则表达式格式
r.system_filename_pattern = '(\d+)_candidate0.txt'

output = r.convert_and_evaluate()

print(output)

output_dict = r.output_to_dict(output)
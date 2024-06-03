from pyrouge import Rouge155

r = Rouge155()

# 测试数据集
r.model_dir = '/data/lihuan/ROUGE/reference'

r.model_filename_pattern = '001_reference.txt'


# 生成数据集
r.system_dir = '/data/lihuan/MedicalQA/3_outputs/output_BART_20231130'

r.system_filename_pattern = '(\d+)_candidate0.txt'



output = r.convert_and_evaluate()

print(output)

output_dict = r.output_to_dict(output)

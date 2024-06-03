import nltk
import jieba
from nltk.translate.bleu_score import sentence_bleu
from LoadDataset import vocab,getAnswers
import json
import csv

def calculate_chinese_bleu_score(reference, candidate):
    """
    计算给定参考和候选句子之间的中文BLEU分数。

    参数：
    reference (list of str): 参考句子的列表，每个句子是一个字符串。
    candidate (str): 候选句子，作为一个字符串。

    返回：
    bleu_score (float): BLEU分数，范围在0到1之间，越高越好。
    """
    # 使用jieba分词将参考句子和候选句子分词
    reference = [list(jieba.cut(reference))]
    candidate = list(jieba.cut(candidate))

    # 计算BLEU分数
    bleu_score = sentence_bleu(reference, candidate)

    return bleu_score

# 用法示例

# 添加医疗词典
vocab=vocab()
for i in  range(len(vocab)):
    jieba.add_word(vocab[i])

refs = []
cans = []

# 加载文件测试集output文件，路径许变更
test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/old/translate/cn_test.json",'r')]

for i in range(len(test_data)):
    
    ref = getAnswers(test_data[i]['ans_contents']).replace(' ','')
    
    if ref != '':
        refs.append(ref)
    else:
        refs.append("无")
    if i%100 == 0:
        print('测试集已加载：' + str(i) + '/5000')

# 调整生成出的答案****************************************************************************

path = '/data/lihuan/MedicalQA/3_outputs/output_BART_20231130' + '/candidate0.csv'

with open(path) as f:
    count = 0
    reader = csv.reader(f)
    for can in reader:
        
        can = can[0].replace(' ','')
        cans.append(can)
        count = count + 1
        
        if count%100 == 0:
            print('生成集已加载：' + str(count) + '/5000')



bleu_score = 0
for i in range(5000):
    bleu_score += calculate_chinese_bleu_score(refs[i], cans[i])

bleu_score = bleu_score/5000
print(f"中文BLEU分数：{bleu_score}")

from nltk.translate.meteor_score import meteor_score
import nltk
from nltk import word_tokenize
import json
import csv
import jieba
from LoadDataset import vocab,getAnswers
# nltk.download('punkt')
# nltk.download('wordnet')

# 添加医疗词典
vocab=vocab()
for i in  range(len(vocab)):
    jieba.add_word(vocab[i])


refs = []  # 给定标准译文
cans = []  # 神经网络生成的句子

# 加载文件测试集output文件，路径许变更
reference_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/old/translate/cn_test.json",'r')]

for i in range(len(reference_data)):
  ref = getAnswers(reference_data[i]['ans_contents']).replace(' ','')
  reference = ((' '.join(jieba.cut(ref))).split())
  refs.append(reference)
  if i%100 == 0:
    print('测试集加载：' + str(i+1) + '/5000')

# 加载生成出的答案****************************************************************************

path = '/data/lihuan/MedicalQA/3_outputs/output_BART_20231130' + '/candidate0.csv'




with open(path) as f:
    reader = csv.reader(f)
    count = 0
    for can in reader:
      
      count = count + 1
      candidate = ((' '.join(jieba.cut(can[0].replace(' ','')))).split())
      cans.append(candidate)
      if count%100 == 0:
        print('生成集加载：' + str(count) + '/5000')




scores = 0
for i in range(5000):
  score = nltk.translate.meteor_score.single_meteor_score(refs[i], cans[i])
  scores = scores + score
  if i%100 == 0:
    print(str(score) + '已评分：' + str(i+1)+ '/5000')


meteor_score = scores/5000
print(meteor_score)



# 计算 METEOR 指标



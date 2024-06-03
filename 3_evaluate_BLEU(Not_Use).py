import jieba
from nltk.translate.bleu_score import sentence_bleu
import json
import csv
from LoadDataset import vocab,getAnswers



hyps = [] # 生成的答案
refs = [] # 参考的答案

# 添加医疗词典
vocab=vocab()
for i in  range(len(vocab)):
    jieba.add_word(vocab[i])


# 加载文件测试集output文件，路径许变更
test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/old/translate/cn_test.json",'r')]

for i in range(len(test_data)):
  ref = getAnswers(test_data[i]['ans_contents']).replace(' ','')
  if ref != '':
    refs.append((' '.join(jieba.cut(ref))))
  else:
    refs.append("无")
  if i%100 == 0:
    print('原始数据已加载：' + str(i) + '/5000')

# 调整生成出的答案****************************************************************************

path = '/data/lihuan/MedicalQA/3_outputs/output_1Q3A_ES_2023_09_20_取所有答案' + '/candidate0.csv'

with open(path) as f:
  count = 0
  reader = csv.reader(f)
  for hyp in reader:
    hyps.append((' '.join(jieba.cut(hyp[0].replace(' ','')))))
    count = count + 1
    if count%100 == 0:
      print('生成数据已加载：' + str(count) + '/5000')


score1 = 0
score2 = 0
score3 = 0
score4 = 0

for i in range(5000):
      reference = []  # 给定标准译文
      candidate = []  # 神经网络生成的句子
      ref = ' '.join(jieba.cut(refs[i]))
      hyp = ' '.join(jieba.cut(hyps[i]))
      reference.append(ref.split())
      candidate = (hyp.split())
      score1 = score1 + sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
      score2 = score2 + sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
      score3 = score3 + sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
      score4 = score4 + sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
      if i%100 == 0:
        print('已完成评估：' + str(i) + '/5000')
# 计算BLEU


reference.clear()
print('Cumulate 1-gram :' + str(score1/5000))
print('Cumulate 2-gram :' + str(score2/5000))
print('Cumulate 3-gram :' + str(score3/5000))
print('Cumulate 4-gram :' + str(score4/5000))

import json
from rouge_chinese import Rouge
import csv
import jieba
from LoadDataset import vocab,getAnswers

# 添加医疗词典
vocab=vocab()
for i in  range(len(vocab)):
    jieba.add_word(vocab[i])



hyps = [] # 生成的答案
refs = [] # 参考的答案
rouge = Rouge() # 初始化Rouge

# 加载文件测试集output文件，路径许变更
test_data = [json.loads(line) for line in open(f"/data/lihuan/MedicalQA/old/translate/loc14_test.json",'r')]

for i in range(len(test_data)):
  ref = getAnswers(test_data[i]['ans_contents']).replace(' ','')
  if ref != '':
    refs.append((' '.join(jieba.cut(ref))))
  else:
    refs.append("无")
  if i%100 == 0:
    print('原始数据已加载：' + str(i) + '/5000')

# 调整生成出的答案****************************************************************************

path = '/data/lihuan/MedicalQA/3_outputs/output_1Q3A_DPR_2023_09_23' + '/candidate0.csv'

with open(path) as f:
  count = 0
  reader = csv.reader(f)
  for line in reader:
    hyp = line[0].replace(' ','')
    hyps.append((' '.join(jieba.cut(hyp))))
    count = count + 1
    if i%100 == 0:
      print('生成数据已加载：' + str(count) + '/5000')

scores = rouge.get_scores(hyps, refs,avg=True)
print(scores)

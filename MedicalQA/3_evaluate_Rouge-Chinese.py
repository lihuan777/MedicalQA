import json
from rouge_chinese import Rouge
import csv
import jieba
from LoadDataset import vocab, getAnswers
from sklearn.metrics import precision_score, recall_score, f1_score

# 添加医疗词典
vocab = vocab()
for i in range(len(vocab)):
    jieba.add_word(vocab[i])

hyps = []  # 生成的答案
refs = []  # 参考的答案
rouge = Rouge()  # 初始化Rouge

# 加载文件测试集output文件，路径许变更
# test_data = [json.loads(line) for line in open(f"D:/code/ROUGE/reference/cMedQA.json", 'r', encoding='utf-8')]

with open("D:/code/ROUGE/reference/001_reference.txt", 'r', encoding='utf-8') as txt_file:
    txt_lines = txt_file.readlines()

for i in range(len(txt_lines)):
    ref = txt_lines[i]
    if ref != '':
        refs.append((' '.join(jieba.cut(ref))))
    else:
        refs.append("无")
    if i % 100 == 0:
        print('原始数据已加载：' + str(i) + '/5000')

# 调整生成出的答案****************************************************************************

path = '3_outputs/2024080201_bert_ES+DPR_1Q3A/candidate0.csv'

with open(path, encoding='utf-8') as f:
    count = 0
    reader = csv.reader(f)
    for line in reader:
        hyp = line[0].replace(' ', '')
        hyps.append((' '.join(jieba.cut(hyp))))
        count = count + 1
        if count % 100 == 0:
            print('生成数据已加载：' + str(count) + '/5000')

# 计算 ROUGE 分数
scores = rouge.get_scores(hyps, refs, avg=True)
print("ROUGE Scores:", scores)

# 计算 F1-score
precisions = []
recalls = []
f1_scores = []

for hyp, ref in zip(hyps, refs):
    hyp_tokens = set(hyp.split())
    ref_tokens = set(ref.split())

    true_positives = len(hyp_tokens & ref_tokens)
    false_positives = len(hyp_tokens - ref_tokens)
    false_negatives = len(ref_tokens - hyp_tokens)

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# 平均 F1-score
average_f1_score = sum(f1_scores) / len(f1_scores)
print("Average F1-Score:", average_f1_score)



import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

# 设置中文字体和 Times New Roman 字体
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置字体顺
plt.rcParams['axes.unicode_minus'] = False   # 解决负号问题

# 设置全局字体大小
plt.rcParams['font.size'] = 16  # 将字体大小设置为 14

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('D:/code/MedicalQA/个性化分类任务/模型/models--google-bert--bert-base-cased')
model = BertModel.from_pretrained('D:/code/MedicalQA/个性化分类任务/模型/models--google-bert--bert-base-cased')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 计算BERT的文本向量表示
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS]位置的向量表示整个句子
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# 绘制t-SNE图
def generate_tsne_pic(file_name, save_name):
    reference_path = "D:/code/ROUGE/reference/cMedQA.txt"
    candidate_path = file_name

    reference_texts = read_file(reference_path)
    candidate_texts = read_file(candidate_path)

    # 获取文本向量表示
    reference_embeddings = get_bert_embeddings(reference_texts)
    candidate_embeddings = get_bert_embeddings(candidate_texts)

    # 将 reference 和 candidate 数据合并
    all_embeddings = np.concatenate([reference_embeddings, candidate_embeddings], axis=0)

    # 使用PCA降维至50维（t-SNE对高维数据表现较好）
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # 使用t-SNE降维至2维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(reduced_embeddings)

    # 绘制t-SNE图
    plt.figure(figsize=(10, 6))

    # 绘制参考文本和候选文本
    plt.scatter(tsne_result[:len(reference_texts), 0], tsne_result[:len(reference_texts), 1], label='参考文本', color='blue')
    plt.scatter(tsne_result[len(reference_texts):, 0], tsne_result[len(reference_texts):, 1], label='候选文本', color='green')

    # 添加图例、标签和标题
    plt.legend(fontsize=16)  # 图例字体大小
    plt.xlabel('t-SNE 第1维', fontsize=16)  # x 轴标签字体大小
    plt.ylabel('t-SNE 第2维', fontsize=16)  # y 轴标签字体大小
    plt.title('t-SNE 可视化图', fontsize=16)  # 图形标题字体大小
    # 显示图形
    plt.grid(True)

    # 保存图片
    plt.savefig(save_name, dpi=300)

if __name__ == '__main__':
    candidate_files = [
        'cMedQA/3_outputs/output_1Q1A_2024102501/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q2A_2024102502/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q3A_2024102503/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q3A_DPR_2024102702/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q3A_ES_2024102701/001_candidate0.txt'
    ]

    save_names = [
        'cMedQA/结果图/tsne_1.png',
        'cMedQA/结果图/tsne_2.png',
        'cMedQA/结果图/tsne_3.png',
        'cMedQA/结果图/tsne_4.png',
        'cMedQA/结果图/tsne_5.png'
    ]

    for i in range(len(candidate_files)):
        print(f"正在处理文件: {candidate_files[i]}")
        generate_tsne_pic(candidate_files[i], save_names[i])
        print(f"图片已保存到: {save_names[i]}")

    print('-------------------------------------------------')

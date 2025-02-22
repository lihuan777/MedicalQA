import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# 设置中文字体和 Times New Roman 字体
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置字体顺
plt.rcParams['axes.unicode_minus'] = False   # 解决负号问题

# 设置全局字体大小
plt.rcParams['font.size'] = 16  # 将字体大小设置为 14

def genarate_pic(file_name, save_name):
    # 检查参考答案文件是否存在
    reference_path = 'D:/code/ROUGE/reference/cMedQA.txt'
    if not os.path.exists(reference_path):
        print(f"参考答案文件不存在: {reference_path}")
        return

    # 检查候选文件是否存在
    if not os.path.exists(file_name):
        print(f"候选文件不存在: {file_name}")
        return

    # 读取参考答案和生成答案
    with open(reference_path, 'r', encoding='utf-8') as ref_file:
        references = ref_file.readlines()

    with open(file_name, 'r', encoding='utf-8') as candidate_file:
        candidates = candidate_file.readlines()

    # 去除空行或仅包含空格的行
    references = [ref.strip() for ref in references if ref.strip()]
    candidates = [candidate.strip() for candidate in candidates if candidate.strip()]

    # 确保参考答案和生成的答案数量一致
    assert len(references) == len(candidates), "参考答案和生成答案的数量不匹配"

    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(stop_words='english')

    # 用于存储误差（1 - cosine similarity）值
    errors = []

    # 计算每一对参考答案和生成答案的Cosine Similarity
    for ref, candidate in zip(references, candidates):
        ref_vec = vectorizer.fit_transform([ref]).toarray()
        candidate_vec = vectorizer.transform([candidate]).toarray()
        
        similarity = cosine_similarity(ref_vec, candidate_vec)[0][0]
        error = 1 - similarity  # 误差 = 1 - Cosine Similarity
        errors.append(error)

    # 绘制误差分析图
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title("文本生成的误差分布（余弦相似度）")
    plt.xlabel("误差 (余弦相似度)")
    plt.ylabel("频率")
    plt.grid(True)

    # 确保保存图片的文件夹存在
    save_dir = os.path.dirname(save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片到文件夹
    plt.savefig(save_name, dpi=300)
    plt.close()  # 关闭当前图表以释放资源

if __name__ == '__main__':

    candidate_files = [
        'cMedQA/3_outputs/output_1Q1A_2024102501/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q2A_2024102502/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q3A_2024102503/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q3A_DPR_2024102702/001_candidate0.txt',
        'cMedQA/3_outputs/output_1Q3A_ES_2024102701/001_candidate0.txt'
    ]

    save_names = [
        'cMedQA/结果图/1-1.png',
        'cMedQA/结果图/1-2.png',
        'cMedQA/结果图/1-3.png',
        'cMedQA/结果图/1-4.png',
        'cMedQA/结果图/1-5.png'
    ]

    for i in range(len(candidate_files)):
        print(f"正在处理文件: {candidate_files[i]}")
        genarate_pic(candidate_files[i], save_names[i])
        print(f"图片已保存到: {save_names[i]}")

    print('-------------------------------------------------')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体和 Times New Roman 字体
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置字体顺
plt.rcParams['axes.unicode_minus'] = False   # 解决负号问题

# 设置全局字体大小
plt.rcParams['font.size'] = 16  # 将字体大小设置为 16




def genarate_pic(file_name, save_name):

    # 读取实际结果文件
    with open("二次检索任务/数据集/reference/reference_3000.txt", "r", encoding="utf-8") as f:
        actual_texts = f.readlines()

    # 读取预测结果文件
    with open(file_name, "r", encoding="utf-8") as f:
        predicted_texts = f.readlines()

    predicted_texts = predicted_texts[:3000]
    actual_texts = actual_texts[:3000]

    # 确保两个文件的长度相同
    assert len(actual_texts) == len(predicted_texts), "文件行数不匹配"

    # 初始化TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english")

    # 计算每对文本之间的余弦相似度
    similarities = []
    for actual, predicted in zip(actual_texts, predicted_texts):
        # 将实际文本和预测文本合并为一个列表
        texts = [actual.strip(), predicted.strip()]
        
        # 将文本转换为TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 计算余弦相似度，注意返回的是一个2x2矩阵，取(0,1)的位置
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarities.append(similarity)

    # 转换为numpy数组进行处理
    similarities = np.array(similarities)

    # 绘制相似度分布图
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, color="skyblue", edgecolor="black")
    plt.title("预测结果与实际结果之间的余弦相似性分布")
    plt.xlabel("余弦相似度")
    plt.ylabel("频率")
    plt.grid(True)

    # 保存图片到指定路径
    plt.savefig(save_name)  # 保存路径可以根据需要调整


if __name__ == '__main__':

    candidate_files = [
        '二次检索任务/数据集/output_1Q3A_DPR_2024013001/001_candidate0.txt',
        '二次检索任务/数据集/output_1Q3A_DPR_20240116/001_candidate0.txt',
        '二次检索任务/数据集/output_1Q1A_2023_09_20_取所有答案/001_candidate0.txt',
        '二次检索任务/数据集/output_1Q3A_DPR_categories_20231016/001_candidate0.txt',
        '二次检索任务/数据集/20231019/001_candidate0.txt',
        '二次检索任务/数据集/20231024/001_candidate0.txt',
        '二次检索任务/数据集/20231023/001_candidate0.txt',
        '二次检索任务/数据集/20230922/001_candidate0.txt'
    ]

    save_names = [
        '二次检索任务/结果图/2-1.png',
        '二次检索任务/结果图/2-2.png',
        '二次检索任务/结果图/2-3.png',
        '二次检索任务/结果图/2-4.png',
        '二次检索任务/结果图/2-5.png',
        '二次检索任务/结果图/2-6.png',
        '二次检索任务/结果图/2-7.png',
        '二次检索任务/结果图/2-8.png'
    ]

    for i in range(len(candidate_files)):
        print(f"正在处理文件: {candidate_files[i]}")
        genarate_pic(candidate_files[i], save_names[i])
        print(f"图片已保存到: {save_names[i]}")

    print('-------------------------------------------------')
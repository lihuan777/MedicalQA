import os
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt



# 设置中文字体和 Times New Roman 字体
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置字体顺
plt.rcParams['axes.unicode_minus'] = False   # 解决负号问题



# 设置全局字体大小
plt.rcParams['font.size'] = 16  # 将字体大小设置为 14




def genarate_pic(file_name, save_name):



    # 设置路径
    reference_path = "二次检索任务/数据集/reference/reference_3000.txt"
    candidate_path = file_name

    # 读取文件内容
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()

    reference_texts = read_file(reference_path)
    candidate_texts = read_file(candidate_path)

    # 初始化 ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 计算 ROUGE 分数
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for ref, cand in zip(reference_texts, candidate_texts):
        ref = ref.strip()  # 去除每行文本的换行符
        cand = cand.strip()
        scores = scorer.score(ref, cand)
        
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    # 计算不同阈值下的平均 ROUGE 分数
    thresholds = range(1, 3001)  # 假设不同阈值为前N条数据
    average_rouge_1 = [sum(rouge_1_scores[:i])/i for i in thresholds]
    average_rouge_2 = [sum(rouge_2_scores[:i])/i for i in thresholds]
    average_rouge_l = [sum(rouge_l_scores[:i])/i for i in thresholds]

    # 绘制 ROUGE 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, average_rouge_1, label='ROUGE-1', color='blue')
    plt.plot(thresholds, average_rouge_2, label='ROUGE-2', color='green')
    plt.plot(thresholds, average_rouge_l, label='ROUGE-L', color='red')

    # 添加图例、标签和标题
    plt.legend(fontsize=16)  # 图例字体大小
    plt.xlabel('生成测试集答案数量', fontsize=16)  # x 轴标签字体大小
    plt.ylabel('ROUGE F-Score', fontsize=16)  # y 轴标签字体大小
    plt.title('ROUGE 曲线', fontsize=16)  # 图形标题字体大小
    # 显示图形
    plt.grid(True)


    
    plt.savefig(save_name, dpi=300)  # 保存图片



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
        '二次检索任务/结果图/3-1.png',
        '二次检索任务/结果图/3-2.png',
        '二次检索任务/结果图/3-3.png',
        '二次检索任务/结果图/3-4.png',
        '二次检索任务/结果图/3-5.png',
        '二次检索任务/结果图/3-6.png',
        '二次检索任务/结果图/3-7.png',
        '二次检索任务/结果图/3-8.png'
    ]

    for i in range(len(candidate_files)):
        print(f"正在处理文件: {candidate_files[i]}")
        genarate_pic(candidate_files[i], save_names[i])
        print(f"图片已保存到: {save_names[i]}")

    print('-------------------------------------------------')
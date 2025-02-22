import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

def evaluation(result_path, test_data_path):
    # 读取预测结果文件并加载内容
    with open(result_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # 提取预测结果和标签列表
    predictions = np.array(output_data['predictions'])
    # labels = output_data['labels']

    # 读取测试数据文件并加载真实标签
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 提取真实标签
    true_labels_str = [item['categories'][0] for item in test_data]

    # 确保预测结果和真实标签数量一致
    if len(predictions) != len(true_labels_str):
        print(f"警告: 预测结果的数量({len(predictions)})与真实标签的数量({len(true_labels_str)})不一致")
    else:
        # 1. 准确率 (Accuracy)
        accuracy = accuracy_score(true_labels_str, predictions)
        # print(f"Accuracy: {accuracy * 100:.2f}")

        # 2. 精确率 (Precision), 召回率 (Recall), F1分数 (F1-Score)
        precision = precision_score(true_labels_str, predictions, average='weighted', labels=np.unique(predictions))
        recall = recall_score(true_labels_str, predictions, average='weighted', labels=np.unique(predictions))
        f1 = f1_score(true_labels_str, predictions, average='weighted', labels=np.unique(predictions))
        # print(f"Precision: {precision * 100:.2f}")
        # print(f"Recall: {recall * 100:.2f}")
        # print(f"F1 Score: {f1 * 100:.2f}")

        # 4. AUC-ROC曲线 (Area Under Curve - Receiver Operating Characteristic)
        # 需要将标签转换为二进制形式进行计算
        true_labels_bin = label_binarize(true_labels_str, classes=np.unique(true_labels_str))
        predictions_bin = label_binarize(predictions, classes=np.unique(true_labels_str))

        # 计算AUC
        fpr, tpr, _ = roc_curve(true_labels_bin.ravel(), predictions_bin.ravel())
        roc_auc = auc(fpr, tpr)
        # print(f"AUC-ROC: {roc_auc * 100:.2f}")

        # 5. PR曲线 (Precision-Recall Curve)
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels_bin.ravel(), predictions_bin.ravel())
        average_precision = average_precision_score(true_labels_bin.ravel(), predictions_bin.ravel())
        # print(f"Average Precision: {average_precision * 100:.2f}")

        # 3. 混淆矩阵 (Confusion Matrix)
        # cm = confusion_matrix(true_labels_str, predictions)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        # plt.title('Confusion Matrix')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()

        # # 绘制ROC曲线
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        # plt.title('Receiver Operating Characteristic')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc='lower right')
        # plt.show()

        # # 绘制PR曲线
        # plt.figure(figsize=(8, 6))
        # plt.plot(recall_curve, precision_curve, color='green', label=f'PR curve (AP = {average_precision:.2f})')
        # plt.title('Precision-Recall Curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.legend(loc='lower left')
        # plt.show()

        print(f"{accuracy * 100:.2f}")
        print(f"{precision * 100:.2f}")
        print(f"{f1 * 100:.2f}")


# 加载结果文件
model_name = '新方法-合并三个最佳模型的结果'

test_data_path = '个性化分类任务/数据集/CMDD/test_data_cleaned.json'
result_path = '个性化分类任务/结果/' + model_name + '/CMDD.json'
# evaluation(result_path, test_data_path)

test_data_path = '个性化分类任务/数据集/Huatuo-26M/test_data_cleaned.json'
result_path = '个性化分类任务/结果/' + model_name + '/Huatuo-26M.json'
evaluation(result_path, test_data_path)

test_data_path = '个性化分类任务/数据集/webMedQA/test_data_cleaned.json'
result_path = '个性化分类任务/结果/' + model_name + '/webMedQA.json'
evaluation(result_path, test_data_path)





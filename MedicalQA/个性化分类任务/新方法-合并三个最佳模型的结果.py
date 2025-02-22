import json
import random

def mix_predict(dataset_name, file_name):
    """
    修改后的合并预测逻辑，根据每个模型的准确度动态选择预测结果。
    
    参数：
    - dataset_name: 数据集名称
    - file_name: 输出文件路径
    """
    # 定义文件路径
    file_paths = [
        '个性化分类任务/结果/glm-4-9b-chat/Q+C/' + dataset_name + '.json',
        '个性化分类任务/结果/BERT-chinese/' + dataset_name + '.json',
        
    ]

    # 初始化一个字典来存储处理后的数据
    temp = {}

    # 遍历文件路径列表
    for i, file_path in enumerate(file_paths):
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 取predictions元素
            predictions = data.get('predictions', [])
            # 将predictions添加到result字典中
            temp[f'predictions{i+1}'] = predictions

    # 获取predictions1, predictions2, predictions3
    predictions1 = temp.get('predictions1', [])
    predictions2 = temp.get('predictions2', [])

    # 初始化新的predictions列表
    new_predictions = []

    

    # 遍历每个元素进行比对
    for i in range(len(predictions1)):
        # 获取每个预测结果中的当前元素
        pred1 = predictions1[i]
        pred2 = predictions2[i]

        # 比对逻辑
        if pred1 == pred2:
            new_predictions.append(pred1)
        elif pred1 == '其他':
            new_predictions.append(pred2)
        else:
            new_predictions.append(pred2)



        

    # 创建新的JSON结构
    new_data = {
        "predictions": new_predictions
    }

    # 将新的JSON数据写入文件
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    print(f"处理完成，新数据已保存到 {file_name}")




dataset_name = 'CMDD'
output_file_path = '个性化分类任务/结果/新方法-合并三个最佳模型的结果/' + dataset_name + '.json'
# 假设每个模型的准确度评分
# mix_predict(dataset_name, output_file_path,)


dataset_name = 'Huatuo-26M'
output_file_path = '个性化分类任务/结果/新方法-合并三个最佳模型的结果/' + dataset_name + '.json'
# 假设每个模型的准确度评分
mix_predict(dataset_name, output_file_path,)

dataset_name = 'webMedQA'
output_file_path = '个性化分类任务/结果/新方法-合并三个最佳模型的结果/' + dataset_name + '.json'
# 假设每个模型的准确度评分
mix_predict(dataset_name, output_file_path,)




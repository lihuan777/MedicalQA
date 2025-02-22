
import json





def cut_dataset(input_file_path,output_file_path):

# Load the JSON file and save the first 3000 entries

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    # Extract the first 3000 entries
    data_3000 = data[:3000]



    # 初始化三个列表，用于存储ans_contents的第1、2、3条数据
    ans_contents1 = []
    ans_contents2 = []
    ans_contents3 = []

    # 遍历每一条数据，提取ans_contents的第一条、第二条和第三条数据
    for item in data_3000:
        if 'ans_contents' in item:
            # 提取第一个元素
            if len(item['ans_contents']) > 0:
                ans_contents1.append(item['ans_contents'][0])
            else:
                ans_contents1.append(None)  # 如果没有第一个元素，添加None
            
            # 提取第二个元素
            if len(item['ans_contents']) > 1:
                ans_contents2.append(item['ans_contents'][1])
            else:
                ans_contents2.append(None)  # 如果没有第二个元素，添加None

            # 提取第三个元素
            if len(item['ans_contents']) > 2:
                ans_contents3.append(item['ans_contents'][2])
            else:
                ans_contents3.append(None)  # 如果没有第三个元素，添加None

    # 将ans_contents1保存到txt文件，这个是原始答案，所以不用保存
    # output_file1 = r'D:\code\MedicalQA\二次检索任务\数据集\2\2 ES 评估1.txt'
    # with open(output_file1, 'w', encoding='utf-8') as outfile1:
    #     for ans in ans_contents1:
    #         outfile1.write(str(ans) + '\n')

    # 将ans_contents2保存到txt文件
    output_file2 = r'D:\code\MedicalQA\二次检索任务\数据集\3\001_candidate0.txt'
    with open(output_file2, 'w', encoding='utf-8') as outfile2:
        for ans in ans_contents2:
            outfile2.write(str(ans) + '\n')

    # 将ans_contents3保存到txt文件
    output_file3 = r'D:\code\MedicalQA\二次检索任务\数据集\3\002_candidate0.txt'
    with open(output_file3, 'w', encoding='utf-8') as outfile3:
        for ans in ans_contents3:
            outfile3.write(str(ans) + '\n')

    print("数据处理完成，已保存到对应的TXT文件。")


# Define the file path and the output path
input_file_path = "D:/code/DPR/DPR-main/outputs/results/translate_BART_1Q3A_train_20230922.json"
output_file_path = "D:/code/MedicalQA/二次检索任务/3 ES+DPR检索数据.json"



cut_dataset(input_file_path,output_file_path)
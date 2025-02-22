import pandas as pd
import json
import re

# 读取CSV文件
questions_df = pd.read_csv('cMedQA/datasets/origin/question.csv')
answers_df = pd.read_csv('cMedQA/datasets/origin/answer.csv')

# 创建一个字典来存储结果
result = {}

# 定义需要删除的字符
chars_to_remove = "①②③④｜′⑤°μＥｔ∩・•℃→‖α∶◆＼＜＞￣↓⑥♪⑦々ᅳ｛｝θ—ｌ●텡↑γ⑧■∽ｉ⒈⒉⒊⒋䃼≡✘ω·＃䧳┲｀⺁＂＊Μā⊙〖〗◎ˇ＿ö＇∅ǔá＆←⑨⑩⃣️ぜしвиδｎ﻿¡ｚ⊥䂳［］ｒ≈ひㄒ━∈※▲ｋ╋±．β／≥≥≤“”、…（）《》〈〉【】〔〕‘’“”\'\"@#\$%\^&\[\]{}\',<>\.\?/\\|`\~\-"

# 定义需要替换的字符映射
num_to_zh_map = {
    'Ｂ': 'B', '―': '-', '－': '-', '×': 'x', '％': '%', '～': '~',
    'Ｃ': 'C', 'Ｋ': 'K', 'ｃ': 'c', 'Ｖ': 'V',
    'ｂ': 'b', '＋': '+', '–': '-', '─': '-', 'Ｈ': 'H', 'Ｉ': 'I',
    '９': '9', '７': '7', '⑴': '1', '⑵': '2', '⑶': '3',
    '⑷': '4', '＝': '=', 'Ｘ': 'X', 'Ｐ': 'P', 'Ｇ': 'G', 'Ｔ': 'T',
    '‰': '%', 'Ⅳ': '5', '⑸': '5', '⑹​': '6', 'Ｍ': 'M', 'ｇ': 'g',
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5',
    '６': '6', '８': '8', '⒏': '8', '⒎': '7', '⑺': '7', '⑻': '8',
    '⑽': '10', '〓': '=', '÷': '/', 'ｅ': 'e', 'Ｓ': 'S', 'Ｕ': 'U',
    '⑼': '9', 'Ｑ': 'Q', 'ａ': 'a', 'Ⅵ': '6', 'Ⅴ': 'V', 'Ｙ': 'Y',
    '➕': '+', 'ｐ': 'p', 'ｖ': 'v', '⒌': '5', '⒍': '6', 'Β': 'B',
    'Ｄ': 'D', 'ｘ': 'x', 'Ａ': 'A', 'Ｏ': 'O', 'Ｒ': 'R', 'ｈ': 'h',
    'ｍ': 'm', 'ｏ': 'o', 'Ｌ': 'L', 'ｄ': 'd', 'Ｎ': 'N', 'Ｗ': 'W',
}

# 定义一个清洗函数
def clean_text(text):
    # 删除不需要的字符
    text = re.sub(f"[{re.escape(chars_to_remove)}]", "", text)
    # 替换映射中的字符
    for old_char, new_char in num_to_zh_map.items():
        text = text.replace(old_char, new_char)
    return text

count = 0

# 遍历answers_df，取出question_id对应的问题和答案
for _, row in answers_df.iterrows():
    question_id = row['question_id']
    answer_content = row['content']
    
    # 对答案进行清洗
    answer_content = clean_text(answer_content)
    
    # 查找对应的question
    question_row = questions_df[questions_df['question_id'] == question_id]
    
    if not question_row.empty:
        question_title = question_row.iloc[0]['content']
        
        # 清洗问题标题
        question_title = clean_text(question_title)
        
        # 如果该问题还没有在result字典中，先添加
        if question_id not in result:
            result[question_id] = {
                'ques_title': question_title,
                'ans_contents': [],
                "categories": [],
                "ans_descriptions": []
            }
        
        # 将答案添加到对应问题的答案列表中
        result[question_id]['ans_contents'].append(answer_content)

    count = count + 1 
    
    if count % 1000 == 0:
        print(f"Processed {count} rows")

# 构建结果列表
processed_data = [entry for entry in result.values()]

# 保存为JSON文件
output_file = 'cMedQA/datasets/cMedQA.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(processed_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON文件已成功保存为 {output_file}")

from transformers import BertForSequenceClassification, BertTokenizer

# 下载模型和分词器
model_name = "hfl/chinese-roberta-wwm-ext"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 保存模型和分词器到本地目录
model.save_pretrained("MedicalQA/2_model/chinese-roberta-wwm-ext")
tokenizer.save_pretrained("MedicalQA/2_model/chinese-roberta-wwm-ext")

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BartForConditionalGeneration
import torch

app = Flask(__name__)

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained("BART/2_model/bart-large-chinese")
model = BartForConditionalGeneration.from_pretrained("BART/2_model/model_1Q3A_DPR_categories_rear_20231023/model0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 解决跨域问题

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    user_input = data['question']
    
    inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"answer": output_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


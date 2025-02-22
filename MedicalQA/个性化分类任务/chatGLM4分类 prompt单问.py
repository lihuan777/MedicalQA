import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("D:\code\MedicalQA\个性化分类任务\模型\glm-4-9b-chat", trust_remote_code=True)

query = '根据文章，选择最合适的医生科室。从以下六个科室中选择，并只输出科室名称：\n内科, 妇产科, 外科, 耳鼻喉科, 皮肤科, 儿科\n请选择一个科室，并仅输出科室名称，不允许有其他输出，只能从六个选项中选择。。文章："ques_title": "近视大概300度。在考虑要不要做OK镜，近视大概300度。在考虑要不要做OK镜，或者近视手术。右眼最近经常疼痛，还流眼泪。这个是怎么回事呢，还能做手术吗。","ques_content": "近视","ans_contents": ["这种情况需要到医院检查排除角膜炎的可能。对于近视300度的情况，暂时不要考虑手术治疗了。建议您尽快到医院进行检查，以确定眼睛疼痛和流眼泪的原因。对于近视300度的情况，可以考虑使用OK镜或者其他非手术治疗方法。"'

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "D:\code\MedicalQA\个性化分类任务\模型\glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))







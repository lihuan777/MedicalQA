from zhipuai import ZhipuAI
client = ZhipuAI(api_key="366109b4d0ea4c579bc6decfc0f7bb74.HxHy1IwoM2NtvEAD")  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4-plus",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": "根据以下医疗问题与医生的回答，请根据问题的症状、诊断或治疗建议，选择最合适的医生科室。问题与医生回答：患者出现头痛、发热症状，医生建议检查血常规。请选择一个科室，并仅输出科室名称。"}
    ],
)
print(response.choices[0].message)
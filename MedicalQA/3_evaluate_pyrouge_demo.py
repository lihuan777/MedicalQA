from pyrouge import Rouge155
from jieba import cut

# 示例摘要和参考摘要
summary = "这是一个示例摘要。"
reference = "这是一个参考摘要。"

# 使用 jieba 进行分词
summary = " ".join(cut(summary))
reference = " ".join(cut(reference))

# 计算 ROUGE 分数
rouge = Rouge155()
score = rouge.score_summary(summary, reference, use_tokenizer=True)

print(score)

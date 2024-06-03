from tqdm import tqdm
import time

# 创建一个示例列表
my_list = list(range(100))

# 使用tqdm包装for循环
for i in tqdm(range(100), desc="Processing"):
    # 模拟一些处理时间
    # time.sleep(0.1)
    a=1

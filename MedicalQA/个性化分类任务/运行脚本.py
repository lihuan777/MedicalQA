import subprocess
import os
import time

# 这里是你的Python文件列表
scripts = [
           '个性化分类任务/BERT-base分类 预测.py', 
           '个性化分类任务/BERT-chinese分类 预测.py', 
           '个性化分类任务/BioBERT分类 预测.py', 
           '个性化分类任务/ERNIE分类 预测.py', 
            
           ]






def run_scripts(scripts):
    for script in scripts:
        print(f"Running {script}...")
        subprocess.run(['python', script], check=True)

def shutdown():
    # Windows关机命令
    os.system('shutdown /s /t 1')

if __name__ == "__main__":
    print("Running Python scripts...")
    run_scripts(scripts)
    print("All scripts have been executed.")
    print("Shutting down the system...")
    # 等待几秒钟，以便你可以看到关机前的消息
    time.sleep(5)
    shutdown()
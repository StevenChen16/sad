# 在train函数中替换现有的加载检查点部分
# 将这些代码粘贴到train.py的相应位置

# 添加导入
from utils.model_adapter import adapt_checkpoint_for_new_architecture

# 在train函数中替换加载检查点的代码
start_epoch = 0
if os.path.exists(latest_checkpoint):
    try:
        # 尝试正常加载检查点
        start_epoch = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
        print(f"{Fore.GREEN}▶ 成功加载检查点，
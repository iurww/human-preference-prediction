import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from dataset.human_preference_dataset import PreferenceDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置本地模型路径
model_path = "./models/deberta"  # 修改为你的实际路径

# 加载 tokenizer 和模型
# 使用 AutoTokenizer 和 AutoModel 会自动识别模型类型
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)
# print(model)

# # 将模型移到 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()  # 设置为评估模式

# # 示例：处理文本
# text = "这是一个测试句子。"

# # Tokenize 输入文本
# inputs = tokenizer(
#     text,
#     return_tensors="pt",  # 返回 PyTorch tensor
#     padding=True,
#     truncation=True,
#     max_length=512,
    
# )

# # 将输入移到 GPU
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # 前向传播
# with torch.no_grad():
#     outputs = model(**inputs)

# print(outputs)
# # 获取输出
# last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
# pooler_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None

# print(f"输入 token IDs 形状: {inputs['input_ids'].shape}")
# print(f"Last hidden state 形状: {last_hidden_state.shape}")
# print(f"Hidden size: {model.config.hidden_size}")
# print(f"设备: {device}")



def _parse_json_field(field):
    """解析 JSON 格式的字段"""
    try:
        parsed = json.loads(field)
        if isinstance(parsed, list):
            return ' '.join(str(x) for x in parsed)
        return str(parsed)
    except:
        return str(field)

LENGTH_FILE = 'data/train_lengths.npy'   # 保存路径
max_len = 0
max_prompt = ""
if os.path.exists(LENGTH_FILE):
    # 直接加载
    lengths = np.load(LENGTH_FILE)
    print('✅ 已加载缓存长度数组')
else:
    # 第一次：重新统计
    print('⏳ 未找到缓存，开始统计 token 长度...')
    train_df = pd.read_csv('data/train_new.csv')
    lengths = []
    cnt = 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc='tokenize'):
        text = (f"Prompt: {_parse_json_field(row['prompt'])}\n\n")
                # f"Response A: {_parse_json_field(row['response_a'])}\n\n"
                # f"Response B: {_parse_json_field(row['response_b'])}")
        tok = tokenizer(text)
        lengths.append(len(tok['input_ids']))
        if len(tok['input_ids']) > max_len:
            max_len = len(tok['input_ids'])
            max_prompt = row
        
    lengths = np.array(lengths)
    # print(cnt)
    np.save(LENGTH_FILE, lengths)          # 落盘
    print('✅ 统计完成并缓存')

print('max length in data :', lengths.max())
print('> 1024 的样本数     :', (lengths > 1024).sum())
print(f"90分位数: {np.percentile(lengths, 90):.0f}")

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# 1. 让 bin 边界落在“整数幂”或“分位点”
#    这里用 Freedman–Diaconis 规则算宽度
q25, q75 = np.percentile(lengths, [25, 75])
bin_width = 2 * (q75 - q25) / len(lengths)**(1/3)
bins = np.arange(min(lengths), max(lengths) + bin_width, bin_width)

# 2. 画直方图
n, _, patches = plt.hist(lengths, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')

# 3. 叠一条核密度曲线
from scipy.stats import gaussian_kde
kde = gaussian_kde(lengths)
x_range = np.linspace(min(lengths), max(lengths), 300)
plt.plot(x_range, kde(x_range)*len(lengths)*bin_width, color='red', lw=2, label='KDE')

plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Sample Length Distribution")
plt.grid(True, ls='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('length_distribution_prompt.png', dpi=300)
print(max_len, max_prompt["prompt"], max_prompt["response_a"], max_prompt["response_b"])
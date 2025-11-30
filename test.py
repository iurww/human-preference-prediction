import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
# from dataset.human_preference_dataset import PreferenceDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


model_path = "./models/deberta" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.to("cuda")



# # ---------------- 两段文本 -----------------
# short_text = "今天天气真好！"                       # 11 个字符
# long_text = "人工智能（Artificial Intelligence，AI）亦称智械爱仕达是大苏打水电工科技阿尔达规范收到回复是绝对符合结束战斗风格和是绝对符合所带来符合器智能，指由人制造出来的机器所表现出来的智能。通常人工智能是指通过普通计算机程序来呈现人类智能的技术。该词也指出研究这样的智能系统是否能够实现，以及如何实现。同时，人类的无数职业也逐渐被其取代。" * 20  # 约 1 300 字

# texts = {"short": short_text, "long": long_text, 'a': short_text, 'b': long_text}

# # ---------------- 主循环 -----------------
# res = {}
# for name, text in texts.items():
#     # 1. tokenize & pad to 1024
#     enc = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding="max_length",
#         max_length=1024
#     ).to("cuda")
#     print(enc['input_ids'].shape, name)   # torch.Size([1, tokens])

#     # 2. 测时：GPU 同步 + Event
#     torch.cuda.synchronize()
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     with torch.no_grad():
#         _ = model(**enc)
#     end.record()
#     torch.cuda.synchronize()
#     elapsed_ms = start.elapsed_time(end)      # 毫秒

#     res[name] = {
#         "tokens": enc.input_ids.shape[1],     # 1024
#         "time_ms": elapsed_ms
#     }
#     print(f"{name:>5} text – tokens: {res[name]['tokens']},  GPU time: {elapsed_ms:.2f} ms")


# def _parse_json_field(field):
#     """解析 JSON 格式的字段"""
#     try:
#         parsed = json.loads(field)
#         if isinstance(parsed, list):
#             return ' '.join(str(x) for x in parsed)
#         return str(parsed)
#     except:
#         return str(field)

# train_df = pd.read_csv('data/test.csv')

# cnt = 0
# for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Analyzing"):
#     cnt += len(json.loads(row['prompt']))

# print("行数：", cnt)
# # print(train_df['winner_model_a'].sum(), train_df['winner_model_b'].sum(), train_df['winner_tie'].sum())

x = torch.Tensor([[0.33,0.33,0.33], [0.33,0.33,0.33]])
label = torch.Tensor([2,0]).long()
print(x, label)
import torch.nn.functional as F
y = F.softmax(x, dim=1)
print(y)
selected = y[range(len(label)), label]
print(selected)
selected_log = torch.log(selected)
print(selected_log)
print(-selected_log.sum() / 2)


loss = F.cross_entropy(x, label)
print(loss)

# train_df = pd.read_csv('data/2feaacdd17022010_split.csv')


# import torch

# samples = torch.load('data/6c7e738a32ba8354_cache.pt')        # List[Dict]

# # special_counts = [s['special_token_counts'] for s in samples]

# # import pandas as pd
# # df_counts = pd.DataFrame(special_counts)       # 每行对应一条样本的统计
# # print(df_counts.head())

# for s in samples:
#     # print(s['input_ids'][0:30])
#     # print((s['input_ids'] == 1).sum())
#     idx = (s['input_ids'] == 3).nonzero(as_tuple=False).flatten()
#     print(idx)      # tensor([ 5, 17, 42])

a = tokenizer.encode("[Response_B]\n", add_special_tokens=False)
print(a)
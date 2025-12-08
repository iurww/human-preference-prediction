import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
# from dataset.human_preference_dataset import PreferenceDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter


model_path = "./models/qwen2.5-1.5b" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.to("cuda")

print(model)



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

# x = torch.Tensor([[0.33,0.33,0.33], [0.33,0.33,0.33]])
# label = torch.Tensor([2,0]).long()
# print(x, label)
# import torch.nn.functional as F
# y = F.softmax(x, dim=1)
# print(y)
# selected = y[range(len(label)), label]
# print(selected)
# selected_log = torch.log(selected)
# print(selected_log)
# print(-selected_log.sum() / 2)


# loss = F.cross_entropy(x, label)
# print(loss)

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

# s = "The world is still half-asleep when you open the curtain, and the sky blushes like it’s embarrassed to be watched. Take a breath—slow enough to taste the coolness on your tongue—and remember that nothing is required of you in this second except to exist. Failures, invoices, and unread messages can wait outside the door; they have no shoes to let themselves in. Give yourself five minutes of mercy, and the day will meet you with softer hands."
# print(len(s))

# a = tokenizer.encode(s + s, add_special_tokens=False)
# print(a, len(a))

# special_ids = set([tokenizer.cls_token_id, 
#                    tokenizer.sep_token_id, 
#                    tokenizer.pad_token_id, 
#                    tokenizer.unk_token_id,
#                    tokenizer.mask_token_id])   # 例如 {0, 1, 2, 3, 50256, ...}
# cols = ['prompt', 'response_a', 'response_b']

# df = pd.read_csv('data/train.csv')
# df[cols] = df[cols].map(lambda s: json.loads(s) if pd.notnull(s) else [])
# assert df[cols].apply(lambda row: len(set(map(len, row))) == 1, axis=1).isin([False]).sum() == 0
# df[cols] = df[cols].map(lambda l: ' '.join([str(x) for x in l]) if len(l) > 0 else '')
# df[cols] = df[cols].map(lambda s: tokenizer.encode(s, add_special_tokens=False))
# # 1. 先拿前 10 行 + cols 列（返回 DataFrame）
# sub_df = df.iloc[:10, df.columns.get_indexer(cols)]

# # 2. 对每个单元格 encode
# encoded = sub_df.applymap(lambda s: tokenizer.encode(s, add_special_tokens=False))


# exit()

# df.to_parquet('tok.parquet', engine='pyarrow', compression='snappy')


# df = pd.read_parquet('tok.parquet')
# df[cols] = df[cols].map(lambda x: [i for i in x if i not in special_ids])
# print(df[cols].head())


# special_ids = sorted(tokenizer.all_special_ids)   # 例如 [0, 1, 2, 50256]

# def count_special_vec(arr):
#     if not isinstance(arr, np.ndarray):
#         return np.zeros(len(special_ids), dtype=int)
#     bc = np.bincount(arr, minlength=max(special_ids)+1)
#     return bc[special_ids].astype(int)

# special_counts = np.vstack(df['response_a'].apply(count_special_vec))

# spec_df = pd.DataFrame(special_counts,
#                        columns=[f'spec_{i}' for i in special_ids],
#                        index=df.index)
# spec_df.columns = [f'{i}({tokenizer.decode([i])})' for i in special_ids]

# print(spec_df.describe())


# special_ids = set(tokenizer.all_special_ids)   # 例如 {0, 1, 2, 3, 50256, ...}

# print("Special IDs:", special_ids)
# def count_special(lst):
#     return sum(1 for tid in lst if tid in special_ids)
# special_counts = df[cols].map(count_special)  
# print(special_counts.describe())


# for row in df.itertuples(index=False):
#     ids = row[4]
#     if len(row[5]) == 0 :
#         print("Empty B:", row[0])
#     if len(row[4]) == 0 :
#         print("Empty A:", row[0])
    # cnt = Counter(ids)
    # special_cnt = {sid: cnt.get(sid, 0) for sid in special_ids}
    # if sum(special_cnt.values()) > 0 and special_cnt[3] > 0:
    #     print(row[0], special_cnt[3])

# lens = df[cols].map(len)  # 计算长度
# lens['total'] = lens.sum(axis=1)    # 计算总长度
# print(lens)
# print(lens.describe())


# # df = df.explode(cols, ignore_index=True)
# # 3. 查看结果
# print(df[cols])

# bad_a = (df['response_a'].isin([False, '']) | df['response_a'].isna()) & ( df['winner_tie'] == 1)
# print("Bad A:", bad_a.sum())
# print(df[bad_a][cols + ['winner_model_a', 'winner_model_b', 'winner_tie']])

# bad_b = (df['response_b'].isin([False, '']) | df['response_b'].isna()) & ( df['winner_model_b'] == 1)
# print("Bad B:", bad_b.sum())
# print(df[bad_b][cols + ['winner_model_a', 'winner_model_b', 'winner_tie']])

# empty = (df[cols].isin([False, '']) | df[cols].isna())
# print(empty.sum())
# mask = empty[['response_a', 'response_b']].any(axis=1)
# print(mask.sum())
# bad_rows = df[mask]
# print(bad_rows[cols + ['winner_model_a', 'winner_model_b', 'winner_tie']])

# mask = (df['response_a'].eq('') | df['response_b'].eq('')) & df['winner_tie'].eq(1)
# bad = df[mask]
# print(bad)





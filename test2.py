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


model_path = "./models/deberta" 
tokenizer = AutoTokenizer.from_pretrained(model_path)

s = "hello world! This is a test."
print(tokenizer.encode(s, add_special_tokens=False))
print(tokenizer.encode("[Prompt]:\n" + s, add_special_tokens=False))
print(tokenizer.encode("[Response_a]:\n" + s, add_special_tokens=False))
print(tokenizer.encode("[Response_b]:\n" + s, add_special_tokens=False))

df = pd.read_csv("./data/train.csv")
cols = ['prompt', 'response_a', 'response_b']

df[cols] = df[cols].map(lambda s: json.loads(s) if pd.notnull(s) else [])

assert df[cols].apply(lambda row: len(set(map(len, row))) == 1, axis=1).isin([False]).sum() == 0

df[cols] = df[cols].map(lambda l: ' '.join([str(x) for x in l]) if len(l) > 0 else '')
df[cols] = df[cols].apply(lambda col: col.apply(lambda s: f"[{col.name.capitalize()}]:\n{s}") )
head = df[cols].head()
print(head)

encoded_head = head.map(lambda s: tokenizer.encode(s, add_special_tokens=False))
print(encoded_head)
# print(head.map(len))
# print(encoded_head.map(len))
print(type(encoded_head.iloc[0]))



# df[cols] = df[cols].map(lambda s: tokenizer.encode(s, add_special_tokens=False))
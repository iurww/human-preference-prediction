import os
import json
import csv
import hashlib
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path
import logging
from collections import Counter


class HumanPreferenceDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        prompt_ratio: float = 0.3,
        cache_dir: str = "./data",
        force_reprocess: bool = False,
        usage: str = "train",
    ):
        """
        Args:
            data: 输入的DataFrame数据
            tokenizer: 分词器
            max_length: 样本最大长度
            prompt_ratio: prompt占总长度的最大比例
            cache_dir: 缓存目录
            force_reprocess: 是否强制重新处理
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_ratio = prompt_ratio
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = usage
        self.force_process = force_reprocess
        self.cache_name = f"{self.cache_dir}/tok_{self.usage}_{len(self.data)}_{self.max_length}.parquet"

        logging.info(f"开始处理数据 {self.usage} 样本数: {len(self.data)} 最大长度: {self.max_length} prompt比例: {self.prompt_ratio}")
        self._process_data()
    
    def _keep_head_tail(self, tokens: List[int], max_len: int) -> List[int]:
        if len(tokens) <= max_len:
            return tokens
        
        keep_len = max_len
        head_len = keep_len // 2
        tail_len = keep_len - head_len
        
        return tokens[:head_len] + tokens[-tail_len:]
    
    def _keep_head(self, tokens: List[int], max_len: int) -> List[int]:
        return tokens[:max_len]
    
    def _keep_tail(self, tokens: List[int], max_len: int) -> List[int]:
        return tokens[-max_len:]
    
    def _merge_and_truncate(self, row: pd.Series, swap: bool = False) -> pd.Series:
        prompt_tokens = row['prompt']
        response_a_tokens = row['response_a']
        response_b_tokens = row['response_b']
        
        # special tokens占用空间 ([CLS], [SEP], [SEP], [SEP])
        special_tokens_count = 4
        available_length = self.max_length - special_tokens_count
        
        # 计算各部分长度限制
        max_prompt_len = int(available_length * self.prompt_ratio)
        actual_prompt_len = min(len(prompt_tokens), max_prompt_len)
        
        # 剩余空间平分给两个response
        remaining_length = available_length - actual_prompt_len
        max_response_a_len = remaining_length // 2
        max_response_b_len = remaining_length - max_response_a_len
        
        # 截断
        prompt_tokens = self._keep_head(prompt_tokens, actual_prompt_len)
        response_a_tokens = self._keep_head(response_a_tokens, max_response_a_len)
        response_b_tokens = self._keep_head(response_b_tokens, max_response_b_len)
        
        # 构建最终序列: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
        # 根据swap参数决定是否交换ab位置
        if swap:
            pos_1_tokens, pos_2_tokens = response_b_tokens, response_a_tokens
        else:
            pos_1_tokens, pos_2_tokens = response_a_tokens, response_b_tokens
            
        input_ids = (
            [self.tokenizer.cls_token_id] +
            prompt_tokens +
            [self.tokenizer.sep_token_id] +
            pos_1_tokens +
            [self.tokenizer.sep_token_id] +
            pos_2_tokens +
            [self.tokenizer.sep_token_id]
        )
        
        if len(input_ids) > self.max_length:
            logging.warning(f"长度超出限制: {len(input_ids)} > {self.max_length}")
            exit(1)
        
        # Padding到max_length
        actual_len = len(input_ids)
        padding_length = self.max_length - actual_len
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        
        return pd.Series({
            'input_ids': input_ids,
            'seq_len': actual_len,
        })


    def _process_data(self):
        data_cols = ['prompt', 'response_a', 'response_b']
        label_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
        special_ids = set([self.tokenizer.cls_token_id, 
                        self.tokenizer.sep_token_id, 
                        self.tokenizer.pad_token_id,                                                                                                                                    
                        self.tokenizer.mask_token_id])

        
        if not self.force_process and os.path.exists(self.cache_name):
            logging.info(f"加载缓存文件: {self.cache_name}")
            self.samples = pd.read_parquet(self.cache_name, engine='pyarrow')
            logging.info(f"缓存文件加载完成，样本数: {len(self.samples)}")
            return
        
        logging.info("处理数据...")
        # 加载cols列中的json字符串
        data_df = self.data[data_cols].map(lambda s: json.loads(s) if pd.notnull(s) else [])
        
        # 确保cols列中每行json数组的长度一致()
        assert data_df.apply(lambda row: len(set(map(len, row))) == 1, axis=1).isin([False]).sum() == 0
        
        # 合并多轮对话
        data_df = data_df.map(lambda l: ' '.join([str(x) for x in l]) if len(l) > 0 else '')
        
        # 清洗数据
        mask = data_df['response_a'].eq(data_df['response_b']) & data_df['winner_tie'].eq(0)
        data_df.loc[mask, ['winner_tie', 'winner_model_a', 'winner_model_b']] = [1, 0, 0]
        
        mask_real_nan_a = (
            data_df['response_a'].isna() |
            data_df['response_a'].str.strip().str.lower().eq("none") |
            data_df['response_a'].str.strip().eq("") |
            data_df['response_a'].astype(str).str.fullmatch(r'\s*')
        )
        mask_real_nan_b = (
            data_df['response_b'].isna() |
            data_df['response_b'].str.strip().str.lower().eq("none") |
            data_df['response_b'].str.strip().eq("") | 
            data_df['response_b'].astype(str).str.fullmatch(r'\s*')
        )
        data_df[mask_real_nan_a & ~mask_real_nan_b & data_df['winner_tie'] == 1][['winner_tie', 'winner_model_a', 'winner_model_b']] = [0, 0, 1]
        data_df[~mask_real_nan_a & mask_real_nan_b & data_df['winner_tie'] == 1][['winner_tie', 'winner_model_a', 'winner_model_b']] = [0, 1, 0]
        data_df[mask_real_nan_a & ~mask_real_nan_b & data_df['winner_model_a'] == 1][['winner_tie', 'winner_model_a', 'winner_model_b']] = [0, 0, 1]
        data_df[mask_real_nan_b & ~mask_real_nan_a & data_df['winner_model_b'] == 1][['winner_tie', 'winner_model_a', 'winner_model_b']] = [0, 1, 0]
        
        # data_df = data_df.apply(lambda col: col.apply(lambda s: f"[{col.name.capitalize()}]:\n{s}") )

        # tokenize
        logging.info("Tokenizing...")
        data_df = data_df.map(lambda s: self.tokenizer.encode(s, add_special_tokens=False))
        
        # 去除special ids
        data_df = data_df.map(lambda x: [i for i in x if i not in special_ids])
        
        # 生成原始顺序的数据
        logging.info("生成原始顺序数据...")
        data_df_original = data_df.apply(lambda row: self._merge_and_truncate(row, swap=False), axis=1)
        
        # 生成交换顺序的数据
        logging.info("生成交换顺序数据...")
        data_df_swapped = data_df.apply(lambda row: self._merge_and_truncate(row, swap=True), axis=1)
        
        # 处理标签
        label_df_original = self.data[label_cols].idxmax(axis=1).map(
            {label_cols[0]: 0, label_cols[1]: 1, label_cols[2]: 2}
        ).to_frame(name='label') 
        
        label_df_swapped = label_df_original.copy()
        label_df_swapped['label'] = label_df_swapped['label'].map(
            {0: 1, 1: 0, 2: 2}
        )
        
        id_df = self.data[['id']]
        
        # 合并原始顺序数据
        df_original = pd.concat([id_df, data_df_original, label_df_original], axis=1)
        df_original['swap'] = False
        
        # 合并交换顺序数据（标签保持不变）
        df_swapped = pd.concat([id_df, data_df_swapped, label_df_swapped], axis=1)
        df_swapped['swap'] = True
        
        # 合并两个数据集
        df = pd.concat([df_original, df_swapped], axis=0, ignore_index=True)
        
        logging.info(f"数据扩充完成，原始行数: {len(df_original)}, 扩充后行数: {len(df)}")
        logging.info(f"保存缓存文件: {self.cache_name}")
        df.to_parquet(self.cache_name, engine='pyarrow', compression='snappy')
        
        self.samples = df
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        input_ids = sample['input_ids']
        label = sample['label']
        id_ = sample['id']
        seq_len = sample['seq_len']

        attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)
        
        return {
            'id': torch.tensor(id_, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    
    import logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    tokenizer = AutoTokenizer.from_pretrained("./models/deberta")
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    dataset = HumanPreferenceDataset(
        data=train_df,
        tokenizer=tokenizer,
        max_length=1024,
        prompt_ratio=0.3,
        cache_dir="./data",
        force_reprocess=False,
        usage="train",
    )
    

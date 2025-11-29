import os
import json
import csv
import hashlib
import sys
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path
import logging


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
        
        # 生成缓存标识
        self.cache_key = self._generate_cache_key()
        self.processed_csv_path = self.cache_dir / f"{self.cache_key}_split.csv"
        self.cache_path = self.cache_dir / f"{self.cache_key}_cache.pt"
        
        # 加载或处理数据
        if not force_reprocess and self.cache_path.exists():
            logging.info(f"从缓存加载: {self.cache_path}")
            self._load_cache()
        else:
            logging.info("处理数据...")
            self._process_data()
            self._save_cache()
    
    def _generate_cache_key(self) -> str:
        key_str = f"{self.max_length}_{self.prompt_ratio}_{self.tokenizer.name_or_path}_{self.usage}_{len(self.data)}"
        logging.info(f"Cache key string: {key_str}")
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _parse_json_field(self, field: str) -> List[str]:
        try:
            return [str(item) if item is not None else "" for item in json.loads(field)]
        except:
            if isinstance(field, list):
                return field
            return [field]
    
    def _expand_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        expanded_rows = []
        
        for _, row in df.iterrows():
            prompts = self._parse_json_field(row['prompt'])
            response_as = self._parse_json_field(row['response_a'])
            response_bs = self._parse_json_field(row['response_b'])
            
            # 保证长度一致
            min_len = min(len(prompts), len(response_as), len(response_bs))
            
            for i in range(min_len):
                new_row = {
                    'id': f"{row['id']}_{i}",
                    'model_a': row['model_a'],
                    'model_b': row['model_b'],
                    'prompt': prompts[i].replace('\n', '\\n'),
                    'response_a': response_as[i].replace('\n', '\\n'),
                    'response_b': response_bs[i].replace('\n', '\\n'),
                    'winner_model_a': row['winner_model_a'],
                    'winner_model_b': row['winner_model_b'],
                    'winner_tie': row['winner_tie']
                }
                expanded_rows.append(new_row)
        
        return pd.DataFrame(expanded_rows)
    
    def _middle_truncate(self, tokens: List[int], max_len: int) -> List[int]:
        """中间截断,保留首尾"""
        if len(tokens) <= max_len:
            return tokens
        
        keep_len = max_len
        head_len = keep_len // 2
        tail_len = keep_len - head_len
        
        return tokens[:head_len] + tokens[-tail_len:]
    
    def _tokenize_and_truncate(
        self, 
        prompt: str, 
        response_a: str, 
        response_b: str
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenize并智能截断
        返回: (input_ids, token_type_ids)
        """
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_a_tokens = self.tokenizer.encode(response_a, add_special_tokens=False)
        response_b_tokens = self.tokenizer.encode(response_b, add_special_tokens=False)
        
        # special tokens占用空间 ([CLS], [SEP], [SEP], [SEP])
        special_tokens_count = 4
        available_length = self.max_length - special_tokens_count
        
        # 计算各部分长度限制
        max_prompt_len = int(available_length * self.prompt_ratio)
        actual_prompt_len = min(len(prompt_tokens), max_prompt_len)
        
        # 剩余空间平分给两个response
        remaining_length = available_length - actual_prompt_len
        max_response_len = remaining_length // 2
        
        # 截断
        prompt_tokens = self._middle_truncate(prompt_tokens, actual_prompt_len)
        response_a_tokens = self._middle_truncate(response_a_tokens, max_response_len)
        response_b_tokens = self._middle_truncate(response_b_tokens, max_response_len)
        
        # 构建最终序列: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
        input_ids = (
            [self.tokenizer.cls_token_id] +
            prompt_tokens +
            [self.tokenizer.sep_token_id] +
            response_a_tokens +
            [self.tokenizer.sep_token_id] +
            response_b_tokens +
            [self.tokenizer.sep_token_id]
        )
        
        if len(input_ids) > self.max_length:
            logging.info(f"长度不匹配: {len(input_ids)}, {self.max_length}")
        
        # 构建token_type_ids (segment embeddings)
        # 0: prompt部分, 1: response_a, 2: response_b
        token_type_ids = (
            [0] * (1 + len(prompt_tokens) + 1) +  # [CLS] + prompt + [SEP]
            [1] * (len(response_a_tokens) + 1) +   # response_a + [SEP]
            [2] * (len(response_b_tokens) + 1)     # response_b + [SEP]
        )
        
        # Padding到max_length
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        token_type_ids += [0] * padding_length
        
        
        return input_ids, token_type_ids
    
    def _process_data(self):
        df = self.data
        
        logging.info("拆分多轮对话样本...")
        expanded_df = self._expand_rows(df)
        
        # logging.info(f"保存拆分后的CSV到: {self.processed_csv_path}")
        # expanded_df.to_csv(
        #     self.processed_csv_path,
        #     index=False,
        #     quoting=csv.QUOTE_MINIMAL,
        #     escapechar='\\',
        #     doublequote=False
        # )
        
        logging.info("Tokenizing...")
        self.samples = []
        
        for _, row in tqdm(expanded_df.iterrows(), total=len(expanded_df), desc="Tokenizing"):
            input_ids, token_type_ids = self._tokenize_and_truncate(
                row['prompt'],
                row['response_a'],
                row['response_b']
            )
            
            if row['winner_model_a'] == 1:
                label = 0
            elif row['winner_model_b'] == 1:
                label = 1
            else:
                label = 2
            
            self.samples.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'attention_mask': torch.tensor(
                    [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids],
                    dtype=torch.long
                ),
                'label': torch.tensor(label, dtype=torch.long),
                'id': row['id'],
                'model_a': row['model_a'],
                'model_b': row['model_b']
            })
        
        logging.info(f"处理完成,共 {len(self.samples)} 个样本")
    
    def _save_cache(self):
        """保存缓存"""
        logging.info(f"保存缓存到: {self.cache_path}")
        torch.save(self.samples, self.cache_path)
    
    def _load_cache(self):
        """加载缓存"""
        self.samples = torch.load(self.cache_path)
        logging.info(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布"""
        labels = [s['label'].item() for s in self.samples]
        return {
            'model_a_win': labels.count(0),
            'model_b_win': labels.count(1),
            'tie': labels.count(2)
        }


if __name__ == "__main__":
    
    import logging, sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    tokenizer = AutoTokenizer.from_pretrained("./models/deberta")
    
    train_df = pd.read_csv('data/train.csv')

    dataset = HumanPreferenceDataset(
        data=train_df,
        tokenizer=tokenizer,
        max_length=1024,
        prompt_ratio=0.3,
        cache_dir="./data",
        force_reprocess=False
    )
    
    print("\n样本示例:")
    sample = dataset[0]
    print(f"ID: {sample['id']}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['label'].item()}")
    print(f"Model A: {sample['model_a']}")
    print(f"Model B: {sample['model_b']}")
    
    print("\n标签分布:")
    dist = dataset.get_label_distribution()
    print(f"Model A 胜: {dist['model_a_win']}")
    print(f"Model B 胜: {dist['model_b_win']}")
    print(f"平局: {dist['tie']}")
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
import json
import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PreferenceDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=1024, cache_dir='./data'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        data_hash = hash(str(len(data)) + str(data.iloc[0].to_dict()))
        self.cache_file = os.path.join(cache_dir, f'tokenized_{data_hash}_{max_length}.pkl')
        
        self.cached_data = self._load_or_create_cache()
    
    def _load_or_create_cache(self):
        """加载或创建tokenize缓存"""
        if os.path.exists(self.cache_file):
            print(f'✅ 加载缓存: {self.cache_file}')
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        
        print('⏳ 缓存不存在,开始tokenize...')
        cached_data = []
        
        from tqdm import tqdm
        for idx in tqdm(range(len(self.data)), desc='Tokenizing'):
            row = self.data.iloc[idx]
            
            # 解析字段
            prompt = self._parse_json_field(row['prompt'])
            response_a = self._parse_json_field(row['response_a'])
            response_b = self._parse_json_field(row['response_b'])
            
            # 分别tokenize三部分
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            response_a_ids = self.tokenizer(response_a, add_special_tokens=False)['input_ids']
            response_b_ids = self.tokenizer(response_b, add_special_tokens=False)['input_ids']
            
            # 智能截断策略
            input_ids, attention_mask = self._smart_truncate(
                prompt_ids, response_a_ids, response_b_ids
            )
            
            # 获取标签
            label = self._get_label(row)
            
            cached_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label
            })
        
        # 保存缓存
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f'✅ 缓存已保存: {self.cache_file}')
        
        return cached_data
    
    def _smart_truncate(self, prompt_ids, response_a_ids, response_b_ids):
        """
        智能截断策略:
        1. 优先保留 prompt (最多截断到 max_length//3)
        2. 平均截断两个 response
        3. 如果还是太长,进一步压缩 prompt
        """
        # 特殊token: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
        # 需要预留 4 个特殊token位置
        max_content_length = self.max_length - 4
        
        total_length = len(prompt_ids) + len(response_a_ids) + len(response_b_ids)
        
        if total_length <= max_content_length:
            # 不需要截断
            pass
        else:
            # 需要截断
            # 策略: prompt保留至少 1/3,两个response平分剩余空间
            prompt_max = max(max_content_length // 3, 128)  # prompt至少保留128个token
            response_max = (max_content_length - min(len(prompt_ids), prompt_max)) // 2
            
            # 截断
            if len(prompt_ids) > prompt_max:
                # 保留前部分prompt
                prompt_ids = prompt_ids[:prompt_max]
            
            if len(response_a_ids) > response_max:
                # 保留前80%和后20%,丢弃中间部分
                keep_head = int(response_max * 0.8)
                keep_tail = response_max - keep_head
                response_a_ids = response_a_ids[:keep_head] + response_a_ids[-keep_tail:]
            
            if len(response_b_ids) > response_max:
                keep_head = int(response_max * 0.8)
                keep_tail = response_max - keep_head
                response_b_ids = response_b_ids[:keep_head] + response_b_ids[-keep_tail:]
        
        # 组合: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
        input_ids = (
            [self.tokenizer.cls_token_id] +
            prompt_ids +
            [self.tokenizer.sep_token_id] +
            response_a_ids +
            [self.tokenizer.sep_token_id] +
            response_b_ids +
            [self.tokenizer.sep_token_id]
        )
        
        attention_mask = [1] * len(input_ids)
        
        return input_ids, attention_mask
    
    def _get_label(self, row):
        """获取标签"""
        if 'winner_model_a' in row:
            if row['winner_model_a'] == 1:
                return 0
            elif row['winner_model_b'] == 1:
                return 1
            else:  # winner_tie == 1
                return 2
        return -1  # 测试集没有标签
    
    def _parse_json_field(self, field):
        """解析 JSON 格式的字段"""
        try:
            parsed = json.loads(field)
            if isinstance(parsed, list):
                return ' '.join(str(x) for x in parsed)
            return str(parsed)
        except:
            return str(field)
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        item = self.cached_data[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def collate_fn(batch):
    """
    自定义 collate 函数,实现动态 padding
    每个 batch padding 到该 batch 内的最大长度
    """
    # 提取各个字段
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    # 动态 padding 到 batch 内最大长度
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels
    }


# 使用示例
if __name__ == '__main__':
    from transformers import AutoTokenizer
    import pandas as pd
    
    # 加载数据和tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./models/deberta')
    train_df = pd.read_csv('data/train_new.csv')
    
    # 创建dataset
    dataset = PreferenceDataset(
        data=train_df,
        tokenizer=tokenizer,
        max_length=1024,
        cache_dir='./data'
    )
    
    # 创建dataloader (使用动态padding)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,  # 关键: 使用自定义collate函数
        num_workers=4
    )
    
    # 测试
    for batch in dataloader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Max length in this batch: {batch['input_ids'].shape[1]}")
        # break
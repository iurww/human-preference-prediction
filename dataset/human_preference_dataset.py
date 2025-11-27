import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class PreferenceDataset(Dataset):
    """人类偏好数据集"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 解析 JSON 格式的字段
        prompt = self._parse_json_field(row['prompt'])
        response_a = self._parse_json_field(row['response_a'])
        response_b = self._parse_json_field(row['response_b'])
        
        # 构建输入文本
        text = f"Prompt: {prompt}\n\nResponse A: {response_a}\n\nResponse B: {response_b}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        # 添加标签（训练集）
        if 'winner_model_a' in row:
            if row['winner_model_a'] == 1:
                label = 0
            elif row['winner_model_b'] == 1:
                label = 1
            else:  # winner_tie == 1
                label = 2
            item['labels'] = torch.tensor(label, dtype=torch.long)
        
        return item
    
    def _parse_json_field(self, field):
        """解析 JSON 格式的字段"""
        try:
            parsed = json.loads(field)
            if isinstance(parsed, list):
                return ' '.join(str(x) for x in parsed)
            return str(parsed)
        except:
            return str(field)



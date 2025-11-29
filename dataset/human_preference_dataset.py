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
        
        # 获取special token IDs
        self.special_token_ids = self._get_special_token_ids()
        
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
    
    def _get_special_token_ids(self) -> Dict[str, int]:
        """获取所有special token的ID"""
        special_tokens = {
            'pad': self.tokenizer.pad_token_id,
            'cls': self.tokenizer.cls_token_id,
            'sep': self.tokenizer.sep_token_id,
            'unk': self.tokenizer.unk_token_id,
            'mask': self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else None,
        }
        # 过滤掉None值
        return {k: v for k, v in special_tokens.items() if v is not None}
    
    def _count_special_tokens(self, input_ids: List[int]) -> Dict[str, int]:
        """统计input_ids中每种special token的数量"""
        counts = Counter(input_ids)
        special_counts = {}
        
        for token_name, token_id in self.special_token_ids.items():
            special_counts[token_name] = counts.get(token_id, 0)
        
        special_counts['total'] = sum(special_counts.values())
        
        return special_counts
    
    def _generate_cache_key(self) -> str:
        key_str = f"{self.max_length}_{self.prompt_ratio}_{self.tokenizer.name_or_path}_{self.usage}_{len(self.data)}"
        logging.info(f"Cache key string: {key_str}")
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除特殊token标记和处理UNK字符"""
        if not isinstance(text, str):
            return ""
        
        # 1. 移除可能的special token文本标记
        special_tokens_to_remove = ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']
        for token in special_tokens_to_remove:
            text = text.replace(token, '')
        
        return text.strip()
        
        # 2. 处理emoji和特殊Unicode字符
        # 可选策略：
        # - 完全移除: 使用下面的代码
        # - 保留: 注释掉下面的代码
        
        # 移除emoji (可选)
        import re
        # emoji pattern
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        
        # 3. 移除其他控制字符，但保留常见标点和空白
        # 保留ASCII可打印字符、中文、日文、韩文等
        def is_valid_char(char):
            code = ord(char)
            # ASCII可打印字符
            if 32 <= code <= 126:
                return True
            # 中文字符
            if 0x4E00 <= code <= 0x9FFF:
                return True
            # 日文平假名和片假名
            if 0x3040 <= code <= 0x30FF:
                return True
            # 韩文
            if 0xAC00 <= code <= 0xD7AF:
                return True
            # 常见标点
            if code in [0x3000, 0x3001, 0x3002, 0xFF0C, 0xFF1A, 0xFF1B, 0xFF1F, 0xFF01]:
                return True
            # 换行和制表符
            if char in ['\n', '\t', '\r']:
                return True
            # 拉丁文扩展（含德语 Ä Ö Ü ß 等）
            if 0x00C0 <= code <= 0x017F:
                return True
            # 西里尔文（俄语、白俄罗斯语、乌克兰语等）
            if 0x0400 <= code <= 0x04FF:
                return True
            return False
        
        text = ''.join(char for char in text if is_valid_char(char))
        
        # 4. 清理多余空白
        # text = ' '.join(text.split())
        
        return text.strip()
    
    def _parse_json_field(self, field) -> List[str]:
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
                # 清理文本
                clean_prompt = self._clean_text(prompts[i])
                clean_response_a = self._clean_text(response_as[i])
                clean_response_b = self._clean_text(response_bs[i])
                
                new_row = {
                    'id': f"{row['id']}_{i}",
                    'model_a': row['model_a'],
                    'model_b': row['model_b'],
                    'prompt': clean_prompt.replace('\n', '\\n'),
                    'response_a': clean_response_a.replace('\n', '\\n'),
                    'response_b': clean_response_b.replace('\n', '\\n'),
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
        
        logging.info("Tokenizing...")
        self.samples = []
        
        # 用于统计所有样本的special token情况
        all_special_counts = []
        
        # 用于记录异常样本
        anomaly_samples = {
            'many_unk': [],
            'many_pad': [],
        }
        
        for idx, row in tqdm(expanded_df.iterrows(), total=len(expanded_df), desc="Tokenizing"):
            input_ids, token_type_ids = self._tokenize_and_truncate(
                row['prompt'],
                row['response_a'],
                row['response_b']
            )
            
            # 统计special token
            special_counts = self._count_special_tokens(input_ids)
            all_special_counts.append(special_counts)
            
            if special_counts['pad'] >= self.max_length - 4:
                continue
            
            # 检测异常情况
            if special_counts['unk'] > 10:
                anomaly_samples['many_unk'].append({
                    'id': row['id'],
                    'unk_count': special_counts['unk'],
                    'prompt': row['prompt'][:100],
                    'response_a': row['response_a'][:100],
                    'response_b': row['response_b'][:100]
                })
            
            if special_counts['pad'] > self.max_length * 0.98:
                anomaly_samples['many_pad'].append({
                    'id': row['id'],
                    'pad_count': special_counts['pad'],
                    'prompt': row['prompt'][:100],
                    'response_a': row['response_a'][:100],
                    'response_b': row['response_b'][:100]
                })
            
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
                'model_b': row['model_b'],
                'special_token_counts': special_counts  # 保存统计信息
            })
        
        # 打印统计结果
        self._print_special_token_stats(all_special_counts)
        
        # 打印异常样本
        # self._print_anomaly_samples(anomaly_samples)
        
        logging.info(f"处理完成,共 {len(self.samples)} 个样本")
    
    def _print_special_token_stats(self, all_special_counts: List[Dict[str, int]]):
        """打印special token统计信息"""
        logging.info("="*60)
        logging.info("Special Token 统计信息")
        logging.info("="*60)
        
        # 计算每种token的平均数量
        token_names = list(self.special_token_ids.keys()) + ['total']
        
        for token_name in token_names:
            counts = [sc[token_name] for sc in all_special_counts]
            avg_count = sum(counts) / len(counts)
            min_count = min(counts)
            max_count = max(counts)
            
            logging.info(f"{token_name.upper():8s}: 平均={avg_count:.2f}, 最小={min_count}, 最大={max_count}")
        
        logging.info("="*60 + "\n")
    
    def _print_anomaly_samples(self, anomaly_samples: Dict[str, List]):
        """打印异常样本信息"""
        logging.info("="*60)
        logging.info("异常样本检测")
        logging.info("="*60)
        
        # 很多UNK
        if anomaly_samples['many_unk']:
            logging.info(f"\n发现 {len(anomaly_samples['many_unk'])} 个样本有大量UNK token (>10):")
            for i, sample in enumerate(anomaly_samples['many_unk'][:3]):
                logging.info(f"  样本ID: {sample['id']}, UNK数量: {sample['unk_count']}")
                logging.info(f"  Prompt前100字符: {sample['prompt']}")
                logging.info(f"  Response A前100字符: {sample['response_a']}")
                logging.info(f"  Response B前100字符: {sample['response_b']}")
        
        if anomaly_samples['many_pad']:
            logging.info(f"\n发现 {len(anomaly_samples['many_pad'])} 个样本有大量PAD token (>99%):")
            for i, sample in enumerate(anomaly_samples['many_pad'][:3]):
                logging.info(f"  样本ID: {sample['id']}, PAD数量: {sample['pad_count']}")
                logging.info(f"  Prompt前100字符: {sample['prompt']}")
                logging.info(f"  Response A前100字符: {sample['response_a']}")
                logging.info(f"  Response B前100字符: {sample['response_b']}")
        
        logging.info("="*60 + "\n")
    
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
    
    def get_special_token_stats(self) -> Dict[str, Dict[str, float]]:
        """获取数据集的special token统计信息"""
        all_counts = [s.get('special_token_counts', {}) for s in self.samples if 'special_token_counts' in s]
        
        if not all_counts:
            return {}
        
        token_names = list(self.special_token_ids.keys()) + ['total']
        stats = {}
        
        for token_name in token_names:
            counts = [sc.get(token_name, 0) for sc in all_counts]
            stats[token_name] = {
                'avg': sum(counts) / len(counts),
                'min': min(counts),
                'max': max(counts),
                'total': sum(counts)
            }
        
        return stats


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
    
    print("\n样本示例:")
    sample = dataset[0]
    print(f"ID: {sample['id']}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['label'].item()}")
    print(f"Model A: {sample['model_a']}")
    print(f"Model B: {sample['model_b']}")
    print(f"Special tokens: {sample.get('special_token_counts', 'N/A')}")
    
    print("\n标签分布:")
    dist = dataset.get_label_distribution()
    print(f"Model A 胜: {dist['model_a_win']}")
    print(f"Model B 胜: {dist['model_b_win']}")
    print(f"平局: {dist['tie']}")
    
    print("\n整体Special Token统计:")
    stats = dataset.get_special_token_stats()
    for token_name, token_stats in stats.items():
        print(f"{token_name.upper()}: {token_stats}")
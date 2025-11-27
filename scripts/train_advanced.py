import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AdvancedPreferenceDataset(Dataset):
    """带特征工程的人类偏好数据集"""
    
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
        
        # 方案1: 分别编码（推荐）
        # 编码 prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length // 3,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码 response_a
        response_a_encoding = self.tokenizer(
            response_a,
            max_length=self.max_length // 3,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码 response_b
        response_b_encoding = self.tokenizer(
            response_b,
            max_length=self.max_length // 3,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 计算额外特征
        features = self._extract_features(prompt, response_a, response_b)
        
        item = {
            'prompt_input_ids': prompt_encoding['input_ids'].squeeze(),
            'prompt_attention_mask': prompt_encoding['attention_mask'].squeeze(),
            'response_a_input_ids': response_a_encoding['input_ids'].squeeze(),
            'response_a_attention_mask': response_a_encoding['attention_mask'].squeeze(),
            'response_b_input_ids': response_b_encoding['input_ids'].squeeze(),
            'response_b_attention_mask': response_b_encoding['attention_mask'].squeeze(),
            'features': torch.tensor(features, dtype=torch.float),
        }
        
        # 添加标签（训练集）
        if 'winner_model_a' in row:
            if row['winner_model_a'] == 1:
                label = 0
            elif row['winner_model_b'] == 1:
                label = 1
            else:
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
    
    def _extract_features(self, prompt, response_a, response_b):
        """提取统计特征"""
        features = []
        
        # 长度特征
        features.append(len(response_a))
        features.append(len(response_b))
        features.append(len(response_a) - len(response_b))
        features.append(len(response_a) / (len(response_b) + 1))
        
        # 词数特征
        words_a = len(response_a.split())
        words_b = len(response_b.split())
        features.append(words_a)
        features.append(words_b)
        features.append(words_a - words_b)
        features.append(words_a / (words_b + 1))
        
        return features


class AdvancedPreferenceModel(nn.Module):
    """高级偏好预测模型"""
    
    def __init__(self, model_name, num_labels=3, num_features=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 特征融合层
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )
    
    def forward(self, prompt_input_ids, prompt_attention_mask,
                response_a_input_ids, response_a_attention_mask,
                response_b_input_ids, response_b_attention_mask,
                features, labels=None):
        
        # 编码 prompt
        prompt_output = self.encoder(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask
        )
        prompt_cls = prompt_output.last_hidden_state[:, 0, :]
        
        # 编码 response_a
        response_a_output = self.encoder(
            input_ids=response_a_input_ids,
            attention_mask=response_a_attention_mask
        )
        response_a_cls = response_a_output.last_hidden_state[:, 0, :]
        
        # 编码 response_b
        response_b_output = self.encoder(
            input_ids=response_b_input_ids,
            attention_mask=response_b_attention_mask
        )
        response_b_cls = response_b_output.last_hidden_state[:, 0, :]
        
        # 特征投影
        feature_emb = self.feature_projection(features)
        
        # 拼接所有特征
        combined = torch.cat([prompt_cls, response_a_cls, response_b_cls, feature_emb], dim=1)
        
        # 分类
        logits = self.classifier(combined)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def compute_log_loss(predictions, labels):
    """计算 Log Loss"""
    n_samples = len(labels)
    y_true = np.zeros((n_samples, 3))
    y_true[np.arange(n_samples), labels] = 1
    
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(predictions)) / n_samples
    
    return loss


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # 移动数据到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = model(
            prompt_input_ids=batch['prompt_input_ids'],
            prompt_attention_mask=batch['prompt_attention_mask'],
            response_a_input_ids=batch['response_a_input_ids'],
            response_a_attention_mask=batch['response_a_attention_mask'],
            response_b_input_ids=batch['response_b_input_ids'],
            response_b_attention_mask=batch['response_b_attention_mask'],
            features=batch['features'],
            labels=batch['labels']
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(batch['labels'].cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    train_log_loss = compute_log_loss(all_preds, all_labels)
    
    return avg_loss, train_log_loss


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(
                prompt_input_ids=batch['prompt_input_ids'],
                prompt_attention_mask=batch['prompt_attention_mask'],
                response_a_input_ids=batch['response_a_input_ids'],
                response_a_attention_mask=batch['response_a_attention_mask'],
                response_b_input_ids=batch['response_b_input_ids'],
                response_b_attention_mask=batch['response_b_attention_mask'],
                features=batch['features'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(batch['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    val_log_loss = compute_log_loss(all_preds, all_labels)
    
    return avg_loss, val_log_loss


def main():
    # 配置
    CONFIG = {
        'model_name': 'microsoft/deberta-v3-base',
        'max_length': 512,
        'batch_size': 4,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42,
        'use_kfold': False,  # 是否使用 K-fold
        'n_splits': 5,
    }
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    wandb.init(
        project='human-preference-prediction',
        config=CONFIG,
        name=f"advanced-deberta-v3-base"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    print('Loading data...')
    train_df = pd.read_csv('data/train.csv')
    
    # 创建标签列用于分层抽样
    train_df['label'] = train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    train_df['label'] = train_df['label'].map({
        'winner_model_a': 0,
        'winner_model_b': 1,
        'winner_tie': 2
    })
    
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    if not CONFIG['use_kfold']:
        # 简单划分
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(
            train_df,
            test_size=0.1,
            random_state=CONFIG['seed'],
            stratify=train_df['label']
        )
        
        print(f'Train size: {len(train_data)}, Val size: {len(val_data)}')
        
        # 创建模型
        model = AdvancedPreferenceModel(CONFIG['model_name'])
        model.to(device)
        
        # 创建数据加载器
        train_dataset = AdvancedPreferenceDataset(train_data.reset_index(drop=True), tokenizer, CONFIG['max_length'])
        val_dataset = AdvancedPreferenceDataset(val_data.reset_index(drop=True), tokenizer, CONFIG['max_length'])
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
        
        # 优化器
        optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        
        total_steps = len(train_loader) * CONFIG['num_epochs']
        warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        # 训练
        best_val_log_loss = float('inf')
        
        for epoch in range(CONFIG['num_epochs']):
            print(f'\n=== Epoch {epoch + 1}/{CONFIG["num_epochs"]} ===')
            
            train_loss, train_log_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            print(f'Train Loss: {train_loss:.4f}, Train Log Loss: {train_log_loss:.4f}')
            
            val_loss, val_log_loss = evaluate(model, val_loader, device)
            print(f'Val Loss: {val_loss:.4f}, Val Log Loss: {val_log_loss:.4f}')
            
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_log_loss': train_log_loss,
                'val_loss': val_loss,
                'val_log_loss': val_log_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            if val_log_loss < best_val_log_loss:
                best_val_log_loss = val_log_loss
                print(f'New best val log loss: {best_val_log_loss:.4f}')
                
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), 'models/advanced_best_model.pth')
                tokenizer.save_pretrained('models/advanced_best_model')
                
                wandb.run.summary['best_val_log_loss'] = best_val_log_loss
        
        print(f'\nTraining completed! Best val log loss: {best_val_log_loss:.4f}')
    
    wandb.finish()


if __name__ == '__main__':
    main()
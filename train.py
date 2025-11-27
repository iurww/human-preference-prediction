import logging
import os
import pandas as pd
import numpy as np
import torch
import wandb

from dataset.human_preference_dataset import PreferenceDataset

from torch.utils.data import  DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_log_loss(predictions, labels):
    """计算 Log Loss"""
    # predictions: (N, 3) 概率分布
    # labels: (N,) 类别标签
    
    # 转换为 one-hot
    n_samples = len(labels)
    y_true = np.zeros((n_samples, 3))
    y_true[np.arange(n_samples), labels] = 1
    
    # 计算 log loss
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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 保存预测和标签用于计算 log loss
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())
        
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    val_log_loss = compute_log_loss(all_preds, all_labels)
    
    return avg_loss, val_log_loss


def main():
    # 配置
    CONFIG = {
        # 'model_name': 'microsoft/deberta-v3-base',  # 可以换成 deberta-v3-large
        'model_name' : './models/deberta',
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42,
    }
    
    logger.info("=" * 60)
    logger.info("Starting training with configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # 初始化 wandb
    wandb.init(
        project='human-preference-prediction',
        config=CONFIG,
        name=f"deberta-v3-base-lr{CONFIG['learning_rate']}"
    )
    logger.info("WandB initialized successfully")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # 加载数据
    logger.info('Loading training data...')
    train_df = pd.read_csv('data/train.csv')
    logger.info(f'Total samples: {len(train_df)}')
    
    # 划分训练集和验证集
    logger.info('Splitting data into train and validation sets...')
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.1, 
        random_state=CONFIG['seed'],
        stratify=train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    )
    logger.info(f'Train size: {len(train_data)}, Validation size: {len(val_data)}')
    
    # 初始化 tokenizer 和模型
    logger.info(f'Initializing model: {CONFIG["model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=3,
    )

    model.to(device)
    logger.info(f'Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters')
    
    # 创建数据集和数据加载器
    logger.info('Creating datasets and dataloaders...')
    train_dataset = PreferenceDataset(train_data.reset_index(drop=True), tokenizer, CONFIG['max_length'])
    val_dataset = PreferenceDataset(val_data.reset_index(drop=True), tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=10
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f'Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}')
    
    # 优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps}')
    
    # 训练循环
    best_val_log_loss = float('inf')
    best_train_log_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f'Epoch {epoch + 1}/{CONFIG["num_epochs"]}')
        logger.info("=" * 60)
        
        # 训练
        train_loss, train_log_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f'[Train] Loss: {train_loss:.4f}, Log Loss: {train_log_loss:.4f}')
        
        # 验证
        val_loss, val_log_loss = evaluate(model, val_loader, device)
        logger.info(f'[Val]   Loss: {val_loss:.4f}, Log Loss: {val_log_loss:.4f}')
        
        # 记录到 wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_log_loss': train_log_loss,
            'val_loss': val_loss,
            'val_log_loss': val_log_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # 保存最佳模型
        if val_log_loss < best_val_log_loss:
        # if train_log_loss < best_train_log_loss:
            # best_train_log_loss = train_log_loss
            best_val_log_loss = val_log_loss
            logger.info(f'🎉 New best validation log loss: {best_val_log_loss:.4f}')
            
            # 保存模型
            os.makedirs('models', exist_ok=True)
            model.save_pretrained('models/best_model')
            tokenizer.save_pretrained('models/best_model')
            logger.info('Model saved to models/best_model/')
            
            wandb.run.summary['best_val_log_loss'] = best_val_log_loss
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f'Training completed! Best validation log loss: {best_val_log_loss:.4f}')
    logger.info("=" * 60)
    wandb.finish()


if __name__ == '__main__':
    main()
"""
使用 torchrun 启动的DDP训练脚本（推荐方式）

运行命令：
torchrun --nproc_per_node=4 train_ddp.py

或指定特定GPU：
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ddp.py
"""

import os
import pandas as pd
import numpy as np
import wandb
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

from configs.logging_config import make_log_dir, init_logger
from configs import CONFIG, print_config
from dataset import HumanPreferenceDataset


def setup_ddp():
    """通过环境变量初始化DDP（torchrun会自动设置这些环境变量）"""
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    return local_rank, dist.get_world_size()


def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    return dist.get_rank() == 0


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    if is_main_process():
        progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}')
    else:
        progress_bar = dataloader
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        if is_main_process():
            wandb.log({'batch_loss': loss.item()})
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if is_main_process():
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # 收集所有进程的loss
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss_tensor.item() / dist.get_world_size()
    
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        if is_main_process():
            progress_bar = tqdm(dataloader, desc='Evaluating')
        else:
            progress_bar = dataloader
            
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    # 收集所有进程的loss
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss_tensor.item() / dist.get_world_size()
    
    return avg_loss


def main():
    # 初始化DDP
    local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # 只在主进程初始化日志和wandb
    if is_main_process():
        init_logger(make_log_dir())
        wandb.init(
            project='human-preference-prediction',
            config=CONFIG,
            name=f"train-deberta-lr{CONFIG['learning_rate']:.1e}-bs{CONFIG['batch_size']}-ep{CONFIG['num_epochs']}-ddp-{world_size}gpus"
        )
        print_config()
        logging.info(f'Using {world_size} GPUs for DDP training')
        logging.info(f'GPU: {torch.cuda.get_device_name(local_rank)}')
        logging.info(f'GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.2f} GB')
    
    # 加载模型和tokenizer
    if is_main_process():
        logging.info(f'Initializing model: {CONFIG["model_name"]}')
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=3,
    )
    
    if CONFIG['use_lora']:
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8, lora_alpha=32, lora_dropout=0.05,
            target_modules=["query_proj", "value_proj"]
        )
        model = get_peft_model(model, peft_config)
        if is_main_process():
            model.print_trainable_parameters()
    
    model.to(device)
    
    # DDP包装
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )
    
    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Trainable parameters: {trainable_params:,}')
    
    # 加载数据
    if is_main_process():
        logging.info('Loading training data...')
    
    train_df = pd.read_csv(CONFIG['train_dataset_path']) if not CONFIG['develop'] else pd.read_csv('data/train_short.csv')
    
    if is_main_process():
        logging.info(f'Total samples: {len(train_df)}')
    
    train_data, val_data = train_test_split(
        train_df, 
        test_size=CONFIG['val_rate'], 
        random_state=CONFIG['seed'],
        stratify=train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    )
    
    if is_main_process():
        logging.info(f'Train size: {len(train_data)}, Validation size: {len(val_data)}')
    
    # 创建数据集
    train_dataset = HumanPreferenceDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        prompt_ratio=CONFIG['prompt_ratio'],
        cache_dir="./data",
        force_reprocess=False,
        usage="train"
    )
    val_dataset = HumanPreferenceDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        prompt_ratio=CONFIG['prompt_ratio'],
        cache_dir="./data",
        force_reprocess=False,
        usage="val"
    )
    
    # DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
        seed=CONFIG['seed']
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=False
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if is_main_process():
        logging.info(f'Train batches per GPU: {len(train_loader)}')
        logging.info(f'Total train batches: {len(train_loader) * world_size}')
        logging.info(f'Effective batch size: {CONFIG["batch_size"] * world_size}')
    
    # 优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    if is_main_process():
        logging.info(f'Total training steps: {total_steps}')
        logging.info(f'Warmup steps: {warmup_steps}')
    
    # 训练循环
    best_train_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        if is_main_process():
            logging.info(f'Epoch {epoch+1}/{CONFIG["num_epochs"]} - Train Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        val_loss = evaluate(model, val_loader, device)
        
        if is_main_process():
            logging.info(f'Epoch {epoch+1}/{CONFIG["num_epochs"]} - Val Loss: {val_loss:.4f}')
            
            wandb.log({
                'epoch': epoch + 1,
                'epoch_avg_train_loss': train_loss,
                'epoch_avg_val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # 保存最佳模型
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                logging.info(f'🎉 New best training loss: {best_train_loss:.4f}')
                
                # 注意：保存DDP模型需要用module
                if CONFIG['use_lora']:
                    model.module.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
                else:
                    model.module.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
                
                tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
                logging.info(f"Best model saved to {CONFIG['checkpoint_dir']}/best_model/")
                
                wandb.run.summary['best_train_loss'] = best_train_loss
            
            # 保存最新模型
            if CONFIG['use_lora']:
                model.module.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
            else:
                model.module.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
            
            tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
        
        # 等待所有进程
        dist.barrier()
    
    if is_main_process():
        logging.info("=" * 60)
        logging.info(f'Training completed! Best training loss: {best_train_loss:.4f}')
        logging.info("=" * 60)
        wandb.finish()
    
    cleanup_ddp()


if __name__ == '__main__':
    main()
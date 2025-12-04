import os
import pandas as pd
import numpy as np
import wandb
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
    """Initialize DDP environment"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, local_rank):
    model.train()
    
    total_loss = 0
    
    # Only show progress bar on main process
    if is_main_process():
        progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}')
    else:
        progress_bar = dataloader
    
    for id_, input_ids, attention_mask, label in progress_bar:
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = label.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Only log on main process
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
    
    # Synchronize loss across all processes
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    
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
    
    # Synchronize loss across all processes
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    
    return avg_loss


def main():
    # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # Only initialize logging and wandb on main process
    if is_main_process():
        init_logger(make_log_dir())
        
        wandb.init(
            project='human-preference-prediction',
            config=CONFIG,
            name=f"train-deberta-ddp\
            -lr{CONFIG['learning_rate']:.1e}\
            -bs{CONFIG['batch_size']}\
            -ep{CONFIG['num_epochs']}\
            -num_gpus{dist.get_world_size()}\
            "
        )
        
        print_config()
        
        logging.info(f'Using DDP with {dist.get_world_size()} GPUs')
        logging.info(f'Local rank: {local_rank}')
        
        if torch.cuda.is_available():
            logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
            logging.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Wait for main process to finish initialization
    if dist.is_initialized():
        dist.barrier()
    
    if is_main_process():
        logging.info(f'Initializing model: {CONFIG["model_name"]}')
    
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'], 
    )
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
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process():
        logging.info(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        logging.info('Loading training data...')
    
    train_df = pd.read_csv(CONFIG['train_dataset_path']) if not CONFIG['develop'] else pd.read_csv('data/train_short.csv')
    
    if is_main_process():
        logging.info(f'Total samples: {len(train_df)}')
        logging.info('Splitting data into train and validation sets...')
    
    train_data, val_data = train_test_split(
        train_df, 
        test_size=CONFIG['val_rate'], 
        random_state=CONFIG['seed'],
        stratify=train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    )
    
    if is_main_process():
        logging.info(f'Train size: {len(train_data)}, Validation size: {len(val_data)}')
        logging.info('Creating datasets and dataloaders...')
    
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
    
    # Create DistributedSampler for train dataset
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=CONFIG['seed']
    )
    
    # Create DistributedSampler for validation dataset
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=val_sampler,
        num_workers=0
    )
    
    if is_main_process():
        logging.info(f'Train batches per GPU: {len(train_loader)}')
        logging.info(f'Validation batches per GPU: {len(val_loader)}')
    
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
        logging.info(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps}, Total epochs: {CONFIG["num_epochs"]}')
    
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        # Set epoch for sampler to shuffle data differently each epoch
        train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, local_rank)
        
        if is_main_process():
            logging.info(f'[Train] Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        val_loss = evaluate(model, val_loader, device)
        
        if is_main_process():
            logging.info(f'[Val] Loss: {val_loss:.4f}')
            
            wandb.log({
                'epoch': epoch + 1,
                'epoch_avg_train_loss': train_loss,
                'epoch_avg_val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                logging.info(f'🎉 New best training loss: {best_train_loss:.4f}')
                
                # Save the underlying model (not the DDP wrapper)
                model.module.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
                tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
                logging.info(f"Model saved to {CONFIG['checkpoint_dir']}/best_model/")
                
                wandb.run.summary['best_train_log_loss'] = best_train_loss
            
            # Save last model
            model.module.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
            tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
            logging.info(f"Model saved to {CONFIG['checkpoint_dir']}/last_model/")
        
        # Synchronize all processes before next epoch
        if dist.is_initialized():
            dist.barrier()
    
    if is_main_process():
        logging.info("")
        logging.info("=" * 60)
        logging.info(f'Training completed! Best training log loss: {best_train_loss:.4f}')
        logging.info("=" * 60)
        
        wandb.finish()
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == '__main__':
    main()
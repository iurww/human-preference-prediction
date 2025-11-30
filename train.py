import os
import pandas as pd
import numpy as np
import wandb
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import  DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

from configs.logging_config import make_log_dir, init_logger
from configs import CONFIG, print_config

from dataset import HumanPreferenceDataset


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}')
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
        wandb.log({'batch_loss': loss.item()})
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
 
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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
    
    return avg_loss


def main():
    init_logger(make_log_dir())

    wandb.init(
        project='human-preference-prediction',
        config=CONFIG,
        name=f"train-deberta-lr{CONFIG['learning_rate']:.1e}-bs{CONFIG['batch_size']}-ep{CONFIG['num_epochs']}"
    )
    
    print_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logging.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    
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
        model.print_trainable_parameters()
    
    model.to(device)
    logging.info(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    logging.info('Loading training data...')
    train_df = pd.read_csv(CONFIG['train_dataset_path']) if not CONFIG['develop'] else pd.read_csv('data/train_short.csv')
    logging.info(f'Total samples: {len(train_df)}')
    
    logging.info('Splitting data into train and validation sets...')
    train_data, val_data = train_test_split(
        train_df, 
        test_size=CONFIG['val_rate'], 
        random_state=CONFIG['seed'],
        stratify=train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    )
    logging.info(f'Train size: {len(train_data)}, Validation size: {len(val_data)}')
    
    logging.info('Creating datasets and dataloaders...')
    train_dataset = HumanPreferenceDataset(
        data=train_df,
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logging.info(f'Train batches: {len(train_loader)}')
    logging.info(f'Validation batches: {len(val_loader)}')
    
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
    
    logging.info(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps}, Total epochs: {CONFIG["num_epochs"]}')
    
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    for epoch in range(CONFIG['num_epochs']):

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logging.info(f'[Train] Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        val_loss = evaluate(model, val_loader, device)
        logging.info(f'[Val] Loss: {val_loss:.4f}')
        
        wandb.log({
            'epoch': epoch + 1,
            'epoch_avg_train_loss': train_loss,
            'epoch_avg_val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # if val_log_loss < best_val_log_loss:
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # best_val_log_loss = val_log_loss
            logging.info(f'🎉 New best training loss: {best_train_loss:.4f}')

            model.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
            tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
            logging.info(f"Model saved to {CONFIG['checkpoint_dir']}/best_model/")
            
            wandb.run.summary['best_train_log_loss'] = best_train_loss
        
        model.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
        tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/last_model")
        logging.info(f"Model saved to {CONFIG['checkpoint_dir']}/last_model/")
    
    logging.info("")
    logging.info("=" * 60)
    logging.info(f'Training completed! Best training log loss: {best_train_loss:.4f}')
    logging.info("=" * 60)
    
    wandb.finish()


if __name__ == '__main__':
    main()
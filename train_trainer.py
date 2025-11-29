import os
import pandas as pd
import numpy as np
import wandb
import logging
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from configs.logging_config import make_log_dir, init_logger
from configs import CONFIG, print_config
from dataset import HumanPreferenceDataset


class WandbCallback(Trainer):
    """自定义 Wandb 回调"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def log(self, logs):
        """重写 log 方法以支持 wandb"""
        super().log(logs)
        if self.state.is_world_process_zero:
            wandb.log(logs)


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    
    return {
        'accuracy': accuracy,
    }


def main():
    init_logger(make_log_dir())

    wandb.init(
        project='human-preference-prediction',
        config=CONFIG,
        name=f"deberta-{CONFIG['learning_rate']}-bs{CONFIG['batch_size']}"
    )
    
    print_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logging.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    
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
        model.print_trainable_parameters()
    
    # model.to(device)
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
    logging.info(f'Train size: {len(train_df)}, Validation size: {len(val_data)}')
    
    logging.info('Creating datasets...')
    train_dataset = HumanPreferenceDataset(
        data=train_df,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        prompt_ratio=CONFIG['prompt_ratio'],
        cache_dir="./data",
        force_reprocess=False,
        usage="train",
    )
    val_dataset = HumanPreferenceDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        prompt_ratio=CONFIG['prompt_ratio'],
        cache_dir="./data",
        force_reprocess=False,
        usage="val",
    )
    
    training_args = TrainingArguments(
        output_dir=CONFIG.get('checkpoint_dir', './checkpoints'),
        
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        
        fp16=CONFIG.get('use_fp16', False),
        bf16=CONFIG.get('use_bf16', False),
        
        gradient_accumulation_steps=CONFIG.get('gradient_accumulation_steps', 1),
        max_grad_norm=1.0,
        
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        
        logging_dir=f"{CONFIG.get('checkpoint_dir', './checkpoints')}/logs",
        logging_strategy='steps',
        logging_steps=50,
        report_to=['wandb'],
        
        seed=CONFIG['seed'],
        dataloader_num_workers=0,
        remove_unused_columns=False,  # 重要：保留自定义字段
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=CONFIG.get('early_stopping_patience', 3)
            )
        ] if CONFIG.get('use_early_stopping', False) else None,
    )
    
    logging.info("=" * 60)
    logging.info("Starting training...")
    logging.info("=" * 60)
    
    train_result = trainer.train()
    
    logging.info("Saving final model...")
    trainer.save_model(f"{CONFIG.get('checkpoint_dir', './checkpoints')}/final_model")
    tokenizer.save_pretrained(f"{CONFIG.get('checkpoint_dir', './checkpoints')}/final_model")
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("Training completed!")
    logging.info(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
    logging.info(f"Best metric: {trainer.state.best_metric:.4f}")
    logging.info("=" * 60)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logging.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    wandb.finish()


if __name__ == '__main__':
    main()
import os
import pandas as pd
import numpy as np
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


def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def main():
    init_logger(make_log_dir())
    print_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logging.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # 加载模型和tokenizer
    logging.info(f'Initializing model: {CONFIG["model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=3,
    )
    
    # 如果使用LoRA
    if CONFIG['use_lora']:
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["query_proj", "value_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    logging.info(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # 加载数据
    logging.info('Loading training data...')
    train_df = pd.read_csv(CONFIG['train_dataset_path']) if not CONFIG['develop'] else pd.read_csv('data/train_short.csv')
    logging.info(f'Total samples: {len(train_df)}')
    
    # 划分训练集和验证集
    logging.info('Splitting data into train and validation sets...')
    train_data, val_data = train_test_split(
        train_df,
        test_size=CONFIG['val_rate'],
        random_state=CONFIG['seed'],
        stratify=train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    )
    logging.info(f'Train size: {len(train_data)}, Validation size: {len(val_data)}')
    
    # 创建数据集
    logging.info('Creating datasets...')
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
    
    # 计算训练步数
    total_steps = (len(train_dataset) // CONFIG['batch_size']) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    logging.info(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps}')
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=CONFIG['checkpoint_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        
        # 评估和保存策略
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        # 日志设置
        logging_dir=f"{CONFIG['checkpoint_dir']}/logs",
        logging_strategy="steps",
        logging_steps=10,
        
        # 其他设置
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        
        # 禁用wandb(如果不需要)
        report_to="none",  # 如果要用wandb,改为 report_to=["wandb"]
        
        # 设置随机种子
        seed=CONFIG['seed'],
    )
    
    from transformers import default_data_collator
    
    def custom_data_collator(features):
        """
        自定义 collator,移除 'id' 字段并将 'label' 重命名为 'labels'
        """
        # 移除 'id' 字段
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])  # 注意: label -> labels
        }
        return batch
    
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if CONFIG.get('early_stopping', False) else None,
    )
    
    # 开始训练
    logging.info("Starting training...")
    train_result = trainer.train()
    
    # 保存最终模型
    logging.info("Saving final model...")
    trainer.save_model(f"{CONFIG['checkpoint_dir']}/best_model")
    tokenizer.save_pretrained(f"{CONFIG['checkpoint_dir']}/best_model")
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 最终评估
    logging.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logging.info("")
    logging.info("=" * 60)
    logging.info(f'Training completed!')
    logging.info(f'Best training loss: {metrics.get("train_loss", "N/A"):.4f}')
    logging.info(f'Final eval loss: {eval_metrics.get("eval_loss", "N/A"):.4f}')
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
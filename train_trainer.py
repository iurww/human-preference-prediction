import os
from time import time
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
import wandb

from configs.logging_config import make_log_dir, init_logger
from configs import CONFIG, print_config
from dataset import HumanPreferenceDataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DetailedLoggingCallback(TrainerCallback):
    
    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
        self.is_main_process = True
        
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self.is_main_process = state.is_world_process_zero
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        pass
        # if self.is_main_process:
        #     logging.info(f">>> Epoch {state.epoch}/{args.num_train_epochs} 开始")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.is_main_process and metrics:
            logging.info("=" * 80)
            logging.info(f"📊 评估结果 (Epoch {int(state.epoch)}):")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logging.info(f"  {key}: {value:.4f}")
            logging.info("=" * 80)
    
    def on_save(self, args, state, control, **kwargs):
        if self.is_main_process:
            logging.info(f"💾 保存checkpoint到: {args.output_dir}/checkpoint-{state.global_step}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = (predictions == labels).mean()
    
    unique_labels = np.unique(labels)
    class_accuracies = {}
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            class_acc = (predictions[mask] == labels[mask]).mean()
            class_accuracies[f'accuracy_class_{label}'] = class_acc
    
    return {
        "accuracy": accuracy,
        **class_accuracies
    }


def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logging.info("=" * 80)
    logging.info("🔧 模型参数统计:")
    logging.info(f"  总参数: {total_params:,}")
    logging.info(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logging.info(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    logging.info("=" * 80 + "\n")


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def is_main_process():
    rank = int(os.environ.get('RANK', 0))
    return rank == 0


def main():
    # ============ DDP 设置 ============
    use_ddp = CONFIG.get('use_ddp', False)
    rank, world_size, local_rank = setup_ddp()
    is_main = is_main_process()
    
    # 只在主进程初始化logger和打印配置
    if is_main:
        init_logger(make_log_dir())
        print_config()
        wandb.init(
            project='human-preference-prediction',
            config=CONFIG,
            name=f"train-deberta-lr{CONFIG['learning_rate']:.1e}-bs{CONFIG['batch_size']}-ep{CONFIG['num_epochs']}"
        )
    
    # 设置设备
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        
        logging.info(f'World Size: {world_size}')
        for i in range(torch.cuda.device_count()):
            logging.info(f'GPU {i}: {torch.cuda.get_device_name(i)} '
                        f'({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'使用设备: {device}')
        if torch.cuda.is_available():
            logging.info(f'GPU型号: {torch.cuda.get_device_name(0)}')
            logging.info(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # ============ 模型初始化 ============
    if is_main:
        logging.info(f'初始化模型: {CONFIG["model_name"]}')
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
   
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=3,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )   
    
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
    num_layers_to_freeze = 10  
    for i, layer in enumerate(model.deberta.encoder.layer):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    if is_main:
        print_model_info(model)
    
    # ============ 数据加载 ============
    if is_main:
        logging.info('加载训练数据...')
    
    train_df = pd.read_csv(CONFIG['train_dataset_path']) if not CONFIG['develop'] else pd.read_csv('data/train_short.csv')
    
    if is_main:
        logging.info('划分训练集和验证集...')
    
    train_data, val_data = train_test_split(
        train_df,
        test_size=CONFIG['val_rate'],
        random_state=CONFIG['seed'],
        stratify=train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    )
    
    # ============ 创建数据集 ============
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
    
    # ============ 训练配置 ============
    # 计算有效的batch size和steps
    effective_batch_size = CONFIG['batch_size']
    gradient_accumulation_steps = CONFIG.get('gradient_accumulation_steps', 1)
    
    if use_ddp:
        # DDP下的实际batch size = per_device_batch_size * num_gpus * gradient_accumulation_steps
        total_batch_size = effective_batch_size * world_size * gradient_accumulation_steps
    else:
        total_batch_size = effective_batch_size * gradient_accumulation_steps
    
    steps_per_epoch = len(train_dataset) // total_batch_size
    total_steps = steps_per_epoch * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    if is_main:
        logging.info(f'训练步数配置:')
        logging.info(f'  Per device batch size: {effective_batch_size}')
        if use_ddp:
            logging.info(f'  Number of GPUs: {world_size}')
        logging.info(f'  Gradient accumulation steps: {gradient_accumulation_steps}')
        logging.info(f'  Total batch size: {total_batch_size}')
        logging.info(f'  每epoch步数: {steps_per_epoch}')
        logging.info(f'  总训练步数: {total_steps}')
        logging.info(f'  预热步数: {warmup_steps}')
    
    # ============ TrainingArguments 详细配置 ============
    training_args = TrainingArguments(
        output_dir=CONFIG['checkpoint_dir'],
        
        # === 训练配置 ===
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=4.0,
        
        # === DDP配置 ===
        ddp_find_unused_parameters=False,
        ddp_backend='nccl' if use_ddp and torch.cuda.is_available() else None,
        
        # === 评估策略 ===
        eval_strategy="epoch",
        
        # === Checkpoint保存策略 ===
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # === Logging配置 ===
        logging_dir=f"{CONFIG['checkpoint_dir']}/logs",
        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,
        
        # === 混合精度训练 ===
        fp16=CONFIG.get('use_amp', False) and torch.cuda.is_available(),
        # bf16=True,
        
        # === 其他设置 ===
        dataloader_num_workers=CONFIG.get('num_workers', 4),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=CONFIG['seed'],
        
        # === 报告工具 ===
        report_to=["wandb"],
        
        # === 分布式相关 ===
        local_rank=local_rank if use_ddp else -1,  # 重要：告诉Trainer当前进程的local_rank
    )
    
    if is_main:
        if CONFIG.get('use_amp', False):
            logging.info('✓ 启用自动混合精度训练 (AMP)')
        
        if training_args.report_to != "none":
            logging.info(f'✓ 启用实验追踪: {training_args.report_to}')

    # ============ Data Collator ============
    def custom_data_collator(features):
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
        return batch
    
    # ============ 创建Trainer ============
    callbacks = [DetailedLoggingCallback(log_every_n_steps=50)]
    
    if CONFIG.get('early_stopping', False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        if is_main:
            logging.info('✓ 启用早停机制 (patience=3)')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    

    train_result = trainer.train()
    
    # ============ 保存最终模型 (只在主进程) ============
    if is_main:
        logging.info("\n💾 保存最终模型...")
        final_model_dir = f"{CONFIG['checkpoint_dir']}/best_model"
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logging.info(f"✓ 模型已保存到: {final_model_dir}")
        
        # ============ 保存训练指标 ============
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # ============ 训练总结 ============
        logging.info("=" * 80)
        logging.info("🎉 训练完成！")
        logging.info("=" * 80)
        logging.info(f"训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
        logging.info(f"最佳模型: {final_model_dir}")
        logging.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()
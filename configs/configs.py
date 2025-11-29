import argparse
from typing import Any, Dict

DEFAULT_CONFIG = {
    'model_name': './models/deberta',
    'train_dataset_path': './data/train.csv',
    'test_dataset_path': './data/test.csv',
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
    
    'max_length': 1024,
    'prompt_ratio': 0.3,
    
    'use_amp': False,
    'use_lora': False,
    'batch_size': 2,
    'learning_rate': 1e-5,
    'num_epochs': 40,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'seed': 42,
    'val_rate': 0.01
}


def parse_command_line_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='训练配置参数')
    
    parser.add_argument('--model_name', type=str, default=None,
                        help='模型名称或路径')
    parser.add_argument('--train_dataset_path', type=str, default=None,
                        help='训练数据集路径')
    parser.add_argument('--test_dataset_path', type=str, default=None,
                        help='测试数据集路径') 
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志目录')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='检查点目录')
    
    
    parser.add_argument('--max_length', type=int, default=None,
                        help='最大序列长度')
    parser.add_argument('--prompt_ratio', type=float, default=None,
                        help='提示词比例')
    
    parser.add_argument('--use_amp', type=bool, default=None,
                        help='是否使用自动混合精度')
    parser.add_argument('--use_lora', type=bool, default=None,
                        help='是否使用LoRA')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--warmup_ratio', type=float, default=None,
                        help='预热比例')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='权重衰减')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')

    parser.add_argument('--val_rate', type=float, default=None,
                        help='验证集比例')
    
    parser.add_argument('--develop', default=False, action='store_true', 
                        help='是否为开发模式')
    
    
    args = parser.parse_args()
    
    return {k: v for k, v in vars(args).items() if v is not None}


def initialize_config() -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    
    cmd_args = parse_command_line_args()
    config.update(cmd_args)
    
    return config


CONFIG = initialize_config()


def print_config():
    import logging
    logging.info("=" * 60)
    logging.info("Starting training with configuration:")
    for key, value in CONFIG.items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 60)


if __name__ == '__main__':
    print_config()
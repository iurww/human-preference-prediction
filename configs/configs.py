CONFIG = {
    # 'model_name': 'microsoft/deberta-v3-base',  # 可以换成 deberta-v3-large
    'model_name' : './models/deberta',
    'max_length': 1024,
    'batch_size': 2,
    'learning_rate': 1e-5,
    'num_epochs': 40,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'seed': 42,
}

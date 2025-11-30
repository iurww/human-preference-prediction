# Human Preference Prediction

使用 DeBERTa-v3 模型预测人类对 LLM 响应的偏好。

## 项目结构

```
.
├── checkpoints
│   └── best_model
├── configs
│   ├── __init__.py
│   ├── configs.py
│   ├── logging_config.py
│   └── random_seed.py
├── data
│   ├── test.csv
│   ├── train.csv
│   └── train_short.csv
├── dataset
│   ├── __init__.py
│   ├── human_preference_dataset.py
│   └── human_preference_test_dataset.py
├── models 
│   └── deberta
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── spm.model
│       └── tokenizer_config.json
├── scripts
│   ├── analyze.py
│   ├── data_processing.py
│   ├── download_data.py
│   ├── find_dirty.py
│   ├── infrrence.py
│   └── train_advanced.py
├── README.md
├── setup.sh
├── inference.py
└──train.py
```

- checkpoint/保存训练模型
- configs/训练配置
- dataset/数据集处理
- data/数据集-kaggle下载
- models/模型文件-huggingface下载
- train.py训练
- inference推理



## 快速开始

```bash
conda create --name mlg3 python=3.13
conda activate mlg3
pip install torch transformers datasets pandas numpy scikit-learn wandb tqdm kaggle sentencepiece matplotlib

wandb login
```



## 使用方法

### 1. 下载数据

放在data目录下,train.csv/test.csv

### 2. 训练模型

```bash
python train.py
```

**功能特性：**

- ✅ 使用 logging 模块记录所有信息
- ✅ 自动划分训练/验证集（90%/10%）
- ✅ WandB 记录所有训练指标
- ✅ 自动保存最佳模型
- ✅ 生成 train.log 日志文件

**训练日志示例：**
```
2024-01-01 10:00:00 - __main__ - INFO - ============================================================
2024-01-01 10:00:00 - __main__ - INFO - Starting training with configuration:
2024-01-01 10:00:00 - __main__ - INFO -   model_name: microsoft/deberta-v3-base
2024-01-01 10:00:00 - __main__ - INFO -   batch_size: 8
2024-01-01 10:00:00 - __main__ - INFO -   learning_rate: 2e-05
...
2024-01-01 10:00:05 - __main__ - INFO - Using device: cuda
2024-01-01 10:00:05 - __main__ - INFO - GPU: NVIDIA GeForce RTX 3090
2024-01-01 10:00:06 - __main__ - INFO - Train size: 41383, Validation size: 4598
...
2024-01-01 10:15:23 - __main__ - INFO - [Train] Loss: 0.8234, Log Loss: 0.8156
2024-01-01 10:16:45 - __main__ - INFO - [Val]   Loss: 0.7845, Log Loss: 0.7823
2024-01-01 10:16:45 - __main__ - INFO - 🎉 New best validation log loss: 0.7823
```

### 3. 生成预测

```bash
python inference.py
```

**输出日志示例：**
```
2024-01-01 11:00:00 - __main__ - INFO - ============================================================
2024-01-01 11:00:00 - __main__ - INFO - Starting inference
2024-01-01 11:00:00 - __main__ - INFO - ============================================================
2024-01-01 11:00:01 - __main__ - INFO - Using device: cuda
2024-01-01 11:00:02 - __main__ - INFO - Test samples: 11496
...
2024-01-01 11:02:34 - __main__ - INFO - Submission saved to submission.csv
2024-01-01 11:02:34 - __main__ - INFO - ============================================================
2024-01-01 11:02:34 - __main__ - INFO - Prediction Statistics:
2024-01-01 11:02:34 - __main__ - INFO -   Model A wins (avg): 0.3245
2024-01-01 11:02:34 - __main__ - INFO -   Model B wins (avg): 0.4123
2024-01-01 11:02:34 - __main__ - INFO -   Ties (avg): 0.2632
2024-01-01 11:02:34 - __main__ - INFO - ============================================================
```

### 4. 提交到 Kaggle

```bash
kaggle competitions submit -c human-preference -f submission.csv -m "DeBERTa-v3-base submission"
```



## 模型配置

在 `configs/configs.py` 中修改配置：

```python
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
```



## 评估指标

使用 **Log Loss** 进行评估：

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c \in \{A,B,TIE\}} \mathbf{1}(y_i = c) \log p_{i,c}
$$

- $N$: 样本数量
- $y_i$: 真实标签
- $p_{i,c}$: 预测概率

**越小越好** ✅


## 参考资源

- [DeBERTa-v3 论文](https://arxiv.org/abs/2111.09543)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [WandB 文档](https://docs.wandb.ai/)
- [Kaggle 竞赛页面](https://www.kaggle.com/c/human-preference)



## TODO

- amp
- 换模型
- lora
- 数据里的unk怎么处理
- 数据里是否有相同问题不同模型回答
- 长样本怎么处理,截断还是滑动窗口
- 多轮问答拆成单轮问答了,最终概率怎么计算
- 多卡训练



## License

MIT
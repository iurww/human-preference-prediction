# Human Preference Prediction

使用 DeBERTa-v3 模型预测人类对 LLM 响应的偏好。

## 项目结构

```
human-preference/
├── pyproject.toml          # UV 包管理配置
├── requirements.txt        # 依赖列表
├── setup.sh               # 一键设置脚本
├── .gitignore
├── README.md
├── src/
│   └── __init__.py        # 空文件，用于包结构
├── scripts/               # 所有 Python 脚本
│   ├── download_data.py   # 数据下载
│   ├── train.py          # 训练脚本
│   ├── train_advanced.py # 高级训练脚本
│   └── inference.py      # 推理脚本
├── data/                 # 数据目录（自动创建）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/               # 模型目录（自动创建）
│   └── best_model/
└── *.log                 # 训练日志文件
```

## 快速开始

### 方法 1：使用一键脚本（推荐）

```bash
# 1. 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2. 创建必要的文件和目录
mkdir -p src scripts
touch src/__init__.py

# 3. 将所有脚本移到 scripts/ 目录
# (download_data.py, train.py, train_advanced.py, inference.py)

# 4. 运行设置脚本
chmod +x setup.sh
./setup.sh

# 5. 配置 Kaggle API
# 创建 ~/.kaggle/kaggle.json:
# {"username": "your_username", "key": "your_api_key"}

# 6. 下载数据
python scripts/download_data.py

# 7. 训练模型
python scripts/train.py

# 8. 生成预测
python scripts/inference.py
```

### 方法 2：手动安装

```bash
# 1. 创建虚拟环境
uv venv
source .venv/bin/activate

# 2. 创建目录结构
mkdir -p src scripts data models
touch src/__init__.py

# 3. 安装依赖（使用 requirements.txt）
uv pip install -r requirements.txt

# 或者直接安装
uv pip install torch transformers datasets pandas numpy scikit-learn wandb tqdm kaggle

# 4. 配置 Kaggle 和 WandB
# ... (见下文)

# 5. 运行脚本
python scripts/download_data.py
python scripts/train.py
python scripts/inference.py
```

## 配置说明

### 1. Kaggle API 配置

创建 `~/.kaggle/kaggle.json` 文件：

```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

获取 API key：https://www.kaggle.com/settings/account

设置权限（Linux/macOS）：
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 2. WandB 配置

```bash
wandb login
```

输入你的 API key（从 https://wandb.ai/authorize 获取）

## 使用方法

### 1. 下载数据

```bash
python scripts/download_data.py
```

**输出日志示例：**
```
2024-01-01 10:00:00 - __main__ - INFO - Data directory created/verified
2024-01-01 10:00:01 - __main__ - INFO - Downloading dataset from Kaggle...
2024-01-01 10:00:10 - __main__ - INFO - Dataset downloaded successfully
2024-01-01 10:00:11 - __main__ - INFO - Dataset extracted successfully
============================================================
Data files:
  - train.csv (45.23 MB)
  - test.csv (11.34 MB)
  - sample_submission.csv (0.89 MB)
============================================================
```

### 2. 训练模型

```bash
python scripts/train.py
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
python scripts/inference.py
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

在 `scripts/train.py` 中修改配置：

```python
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',  # 模型名称
    'max_length': 512,                          # 最大序列长度
    'batch_size': 8,                            # 批次大小
    'learning_rate': 2e-5,                      # 学习率
    'num_epochs': 3,                            # 训练轮数
    'warmup_ratio': 0.1,                        # 预热比例
    'weight_decay': 0.01,                       # 权重衰减
    'seed': 42,                                 # 随机种子
}
```

### 推荐配置

**快速实验**（需要 8GB GPU）：
```python
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',
    'batch_size': 8,
    'num_epochs': 3,
}
```

**高性能**（需要 16GB+ GPU）：
```python
CONFIG = {
    'model_name': 'microsoft/deberta-v3-large',
    'batch_size': 4,
    'max_length': 768,
    'num_epochs': 5,
}
```

**CPU 训练**：
```python
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',
    'batch_size': 2,
    'num_epochs': 1,
}
```

## 日志系统

项目使用 Python logging 模块，所有日志会：
- 输出到控制台
- 保存到 `train.log` 文件

**日志级别：**
- INFO: 正常运行信息
- WARNING: 警告信息
- ERROR: 错误信息

## WandB 监控指标

训练过程中自动记录：

| 指标 | 说明 |
|------|------|
| `train_loss` | 训练损失 |
| `train_log_loss` | 训练集 Log Loss |
| `val_loss` | 验证损失 |
| `val_log_loss` | 验证集 Log Loss（主要指标）⭐ |
| `learning_rate` | 当前学习率 |
| `best_val_log_loss` | 最佳验证 Log Loss |

在 WandB 网页端查看：https://wandb.ai/

## 评估指标

使用 **Log Loss** 进行评估：

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c \in \{A,B,TIE\}} \mathbf{1}(y_i = c) \log p_{i,c}
$$

- $N$: 样本数量
- $y_i$: 真实标签
- $p_{i,c}$: 预测概率

**越小越好** ✅

## 常见问题

### Q1: UV 安装失败

**错误：** `Failed to build human-preference-prediction`

**解决方案：**
```bash
# 不使用 -e 模式，直接安装依赖
mkdir -p src
touch src/__init__.py
uv pip install -r requirements.txt
```

### Q2: GPU 内存不足

**错误：** `CUDA out of memory`

**解决方案：**
```python
# 在 train.py 中调整
CONFIG = {
    'batch_size': 4,  # 或更小，如 2
    'max_length': 256,  # 减小序列长度
}
```

### Q3: Kaggle 下载失败

**错误：** `403 Forbidden`

**解决方案：**
1. 检查 `~/.kaggle/kaggle.json` 是否正确
2. 在 Kaggle 网站上接受比赛规则
3. 检查 API key 是否过期

### Q4: 训练速度慢

**解决方案：**
- 使用 GPU（CUDA）而不是 CPU
- 使用 `deberta-v3-base` 而不是 `large`
- 增加 `batch_size`（如果显存允许）
- 减小 `max_length`

### Q5: WandB 无法登录

**解决方案：**
```bash
# 重新登录
wandb login --relogin

# 或使用环境变量
export WANDB_API_KEY=your_api_key
```

## 性能优化建议

### 1. 数据增强
```python
# 交换 response_a 和 response_b
# 在 PreferenceDataset 中实现
```

### 2. 模型集成
```bash
# 训练多个模型
python scripts/train.py --seed 42
python scripts/train.py --seed 123
python scripts/train.py --seed 456

# 预测时取平均
```

### 3. 更长的序列
```python
CONFIG = {
    'max_length': 768,  # 从 512 增加到 768
}
```

### 4. 学习率调优
```python
# 尝试不同的学习率
learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
```

## 项目特点

- ✅ 使用 UV 包管理
- ✅ 使用 logging 模块记录日志（不使用 print）
- ✅ WandB 完整集成
- ✅ 自动保存最佳模型
- ✅ 完整的错误处理
- ✅ 详细的日志输出
- ✅ 支持 GPU/CPU 训练
- ✅ 代码结构清晰

## 参考资源

- [DeBERTa-v3 论文](https://arxiv.org/abs/2111.09543)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [WandB 文档](https://docs.wandb.ai/)
- [Kaggle 竞赛页面](https://www.kaggle.com/c/human-preference)

## License

MIT
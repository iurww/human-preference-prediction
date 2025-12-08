import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from huggingface_hub import snapshot_download

from configs.configs import CONFIG

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 你的目标模型
MODEL_CACHE_DIR = "./models/qwen2.5-1.5b"
MAX_LENGTH = 2048   # 根据显存调整，建议 2048 或 4096
BATCH_SIZE = 4      # 根据显存调整
LR = 2e-4           # LoRA 常用学习率
NUM_EPOCHS = 3      # 训练轮数
LORA_RANK = 16      # LoRA 的秩
LORA_ALPHA = 32     # LoRA 的缩放因子
LORA_DROPOUT = 0.05

# ==========================================
# 2. 数据集构建 (Dataset Class)
# ==========================================
class PreferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 核心：构造输入文本 ---
        # 建议格式：用户指令 + A回答 + B回答
        # 注意：这里使用特殊标记分隔，帮助模型区分部分
        input_text = (
            f"User Prompt:\n{row['prompt']}\n\n"
            f"### Model A Response:\n{row['response_a']}\n\n"
            f"### Model B Response:\n{row['response_b']}"
        )
        
        # --- 标签处理 ---
        # 将 One-Hot 转换为类别索引 (0, 1, 2)
        # 0: A wins, 1: B wins, 2: Tie
        if row['winner_model_a'] == 1:
            label = 0
        elif row['winner_model_b'] == 1:
            label = 1
        else:
            label = 2 # Tie

        # --- Tokenization ---
        # 截断策略：通常保留 Response 完整更重要，所以从左侧截断 Prompt
        # Qwen 的 Tokenizer 默认 padding_side='right'，这里可以保持
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. 加载模型与 Tokenizer
# ==========================================
print("Loading Tokenizer and Model...")
# local_model_path = snapshot_download(
#     repo_id=MODEL_NAME, 
#     local_dir=MODEL_CACHE_DIR
# )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE_DIR,
    trust_remote_code=True,
)
print(tokenizer.pad_token_id)

# 加载分类模型 (注意 num_labels=3)
# device_map="auto" 会自动利用 GPU
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CACHE_DIR,
    num_labels=3, 
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

# ==========================================
# 4. 配置 LoRA (LoRA Setup)
# ==========================================
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # 任务类型：序列分类
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    # 针对 Qwen2.5 的全模块微调效果最好
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 打印可训练参数量
print(model)
exit()

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=CONFIG['weight_decay']
)

long_text = "人工智能（Artificial Intelligence，AI）亦称智械爱仕达是大苏打水电工科技阿尔达规范收到回复是绝对符合结束战斗风格和是绝对符合所带来符合器智能，指由人制造出来的机器所表现出来的智能。通常人工智能是指通过普通计算机程序来呈现人类智能的技术。该词也指出研究这样的智能系统是否能够实现，以及如何实现。同时，人类的无数职业也逐渐被其取代。" * 20  # 约 1 300 字
t = tokenizer(long_text, return_tensors="pt",  padding="max_length", max_length=1024, truncation=True).to("cuda")

print(t['input_ids'].shape, t['input_ids'])
outout = model(**t, labels=torch.tensor([0]).to("cuda")  )
print(outout)
outout.loss.backward()
optimizer.step()
exit()
# ==========================================
# 5. 准备数据 (Prepare Data)
# ==========================================
# 假设你的 DataFrame 变量名叫 df_train
df_train = pd.read_csv("your_processed_train.csv") 

# 简单切分验证集
train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)

train_dataset = PreferenceDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = PreferenceDataset(val_df, tokenizer, MAX_LENGTH)

# ==========================================
# 6. 定义评估指标 (Optional Metric)
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 计算 LogLoss 需要 softmax 后的概率，但这里为了简单先看 Accuracy
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# ==========================================
# 7. 训练器配置 (Trainer Setup)
# ==========================================
training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-preference-output",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=False,      # 如果上面用了 bfloat16，这里设为 False
    bf16=True,       # 30系/40系显卡建议开启 bf16
    gradient_accumulation_steps=4, # 显存不够时增加这个值，减小 batch_size
    warmup_ratio=0.03,
    optim="adamw_torch",
    report_to="none" # 或者 "wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ==========================================
# 8. 开始训练 (Start Training)
# ==========================================
print("Starting training...")
trainer.train()

# 保存最终模型
trainer.save_model("./final_model")
print("Training finished.")

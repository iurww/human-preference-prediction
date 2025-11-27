import torch
from transformers import AutoTokenizer, AutoModel

# 设置本地模型路径
model_path = "./models/deberta"  # 修改为你的实际路径

# 加载 tokenizer 和模型
# 使用 AutoTokenizer 和 AutoModel 会自动识别模型类型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 将模型移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # 设置为评估模式

# 示例：处理文本
text = "这是一个测试句子。"

# Tokenize 输入文本
inputs = tokenizer(
    text,
    return_tensors="pt",  # 返回 PyTorch tensor
    padding=True,
    truncation=True,
    max_length=512
)

# 将输入移到 GPU
inputs = {k: v.to(device) for k, v in inputs.items()}

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

print(outputs)
# 获取输出
last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
pooler_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None

print(f"输入 token IDs 形状: {inputs['input_ids'].shape}")
print(f"Last hidden state 形状: {last_hidden_state.shape}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"设备: {device}")
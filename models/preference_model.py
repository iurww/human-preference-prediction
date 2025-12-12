import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class MultiLayerMLPClassifier(nn.Module):
    """多层 MLP 分类器"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建多层隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class Qwen25TextClassifier(nn.Module):
    """Qwen2.5-1.5B + LoRA + 多层MLP分类器"""
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B",
        num_classes=3,
        mlp_hidden_dims=[512, 256],
        mlp_dropout=0.1,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        pooling_method="mean"  # "mean", "max", "cls", "last"
    ):
        super().__init__()
        
        self.pooling_method = pooling_method
        
        # 加载 Qwen2.5 模型
        print(f"Loading model: {model_name}")
        self.qwen = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 配置 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 使用特征提取模式
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        # 应用 LoRA
        self.qwen = get_peft_model(self.qwen, peft_config)
        
        # 获取隐藏层维度
        hidden_size = self.qwen.config.hidden_size
        
        # 多层 MLP 分类器
        self.classifier = MultiLayerMLPClassifier(
            input_dim=hidden_size,
            hidden_dims=mlp_hidden_dims,
            num_classes=num_classes,
            dropout=mlp_dropout
        )
        
        print(f"Model created with {num_classes} classes")
        print(f"Hidden size: {hidden_size}")
        print(f"MLP architecture: {hidden_size} -> {' -> '.join(map(str, mlp_hidden_dims))} -> {num_classes}")
        self.qwen.print_trainable_parameters()
    
    def pool_output(self, hidden_states, attention_mask):
        """对序列进行池化操作"""
        if self.pooling_method == "cls":
            # 使用 [CLS] token (第一个token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_method == "last":
            # 使用最后一个有效 token
            batch_size = hidden_states.size(0)
            sequence_lengths = attention_mask.sum(dim=1) - 1
            return hidden_states[torch.arange(batch_size), sequence_lengths, :]
        
        elif self.pooling_method == "max":
            # Max pooling
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float('-inf')
            )
            return torch.max(hidden_states, dim=1)[0]
        
        else:  # mean pooling (默认)
            # Mean pooling
            sum_embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask, labels=None):
        # 通过 Qwen2.5 编码器
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state
        
        # 池化
        pooled_output = self.pool_output(hidden_states, attention_mask)
        
        # 通过 MLP 分类器
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': pooled_output
        }


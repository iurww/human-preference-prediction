import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiLayerClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=3, dropout=0.3):
        super().__init__()
        
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size // 2, num_labels)
        
        self.activation = nn.GELU()
        
    def forward(self, pooled_output):
        x = self.dense1(pooled_output)
        x = self.activation(x)
        x = self.dropout1(x)
        logits = self.dense2(x)
        return logits


class AdvancedClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=3, dropout=0.3):
        super().__init__()
        
        mid_size = hidden_size // 2
        small_size = hidden_size // 4
        
        self.dense1 = nn.Linear(hidden_size, mid_size)
        self.dense2 = nn.Linear(mid_size, small_size)
        self.dense3 = nn.Linear(small_size, num_labels)
        
        self.shortcut = nn.Linear(hidden_size, small_size)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(small_size)
        
    def forward(self, pooled_output):
        x = self.dense1(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.dense2(x)
        x = self.activation(x)
        
        shortcut = self.shortcut(pooled_output)
        x = x + shortcut
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        logits = self.dense3(x)
        return logits


class AttentionPoolingClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=3, dropout=0.3):
        super().__init__()
        
        # 注意力池化
        self.attention = nn.Linear(hidden_size, 1)
        
        # 分类头
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, num_labels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, sequence_output, attention_mask=None):
        # sequence_output: [batch, seq_len, hidden_size]
        
        # 计算注意力权重
        attention_scores = self.attention(sequence_output)  # [batch, seq_len, 1]
        
        if attention_mask is not None:
            # 将padding位置的注意力设为-inf
            attention_mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            attention_scores = attention_scores.masked_fill(
                attention_mask_expanded == 0, float('-inf')
            )
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
        
        # 加权平均
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)  # [batch, hidden_size]
        pooled_output = self.layer_norm(pooled_output)
        
        # 分类
        x = self.dense1(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        
        return logits


class CustomDebertaForSequenceClassification(PreTrainedModel):
    """
    使用DeBERTa backbone + 自定义分类头
    """
    def __init__(self, config, classifier_type='multi_layer', dropout=0.3):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # 加载预训练的DeBERTa（不含分类头）
        self.deberta = AutoModel.from_pretrained(
            config._name_or_path,
            config=config,
            add_pooling_layer=False  # 不使用默认的pooler
        )
        
        # 自定义分类头
        hidden_size = config.hidden_size
        
        if classifier_type == 'multi_layer':
            self.classifier = MultiLayerClassifier(
                hidden_size, 
                num_labels=config.num_labels, 
                dropout=dropout
            )
        elif classifier_type == 'advanced':
            self.classifier = AdvancedClassifier(
                hidden_size, 
                num_labels=config.num_labels, 
                dropout=dropout
            )
        elif classifier_type == 'attention':
            self.classifier = AttentionPoolingClassifier(
                hidden_size, 
                num_labels=config.num_labels, 
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
        self.classifier_type = classifier_type
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # DeBERTa forward
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取输出
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # 根据分类器类型选择pooling方式
        if self.classifier_type == 'attention':
            logits = self.classifier(sequence_output, attention_mask)
        else:
            # 使用[CLS] token的输出
            pooled_output = sequence_output[:, 0, :]  # [batch, hidden_size]
            logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============ 在train_trainer.py中使用 ============
def create_custom_model(model_name, classifier_type='multi_layer', dropout=0.3):
    """
    创建自定义模型的辅助函数
    
    Args:
        model_name: 预训练模型路径
        classifier_type: 'multi_layer', 'advanced', 或 'attention'
        dropout: dropout比例
    """
    from transformers import AutoConfig
    
    # 加载配置
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 3
    config._name_or_path = model_name  # 保存模型路径
    
    # 创建模型
    model = CustomDebertaForSequenceClassification(
        config=config,
        classifier_type=classifier_type,
        dropout=dropout
    )
    
    return model


# ============ 使用示例 ============
if __name__ == '__main__':
    # 测试代码
    from transformers import AutoTokenizer
    
    model_name = "./models/deberta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建模型
    model = create_custom_model(
        model_name=model_name,
        classifier_type='advanced',  # 选择: 'multi_layer', 'advanced', 'attention'
        dropout=0.3
    )
    
    # 测试前向传播
    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    outputs = model(**inputs)
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits: {outputs.logits}")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Classifier ratio: {classifier_params/total_params*100:.2f}%")
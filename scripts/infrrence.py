import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict(model, dataloader, device):
    """生成预测结果"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
    
    return np.vstack(all_preds)


def main():
    # 配置
    model_path = 'models/best_model'
    test_file = 'data/test.csv'
    output_file = 'submission.csv'
    batch_size = 16
    max_length = 512
    
    logger.info("=" * 60)
    logger.info("Starting inference")
    logger.info("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f'Model not found at {model_path}')
        logger.error('Please train the model first by running: python train.py')
        return
    
    if not os.path.exists(test_file):
        logger.error(f'Test file not found at {test_file}')
        logger.error('Please download the data first by running: python download_data.py')
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载测试数据
    logger.info(f'Loading test data from {test_file}...')
    test_df = pd.read_csv(test_file)
    logger.info(f'Test samples: {len(test_df)}')
    
    # 加载模型
    logger.info(f'Loading model from {model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    logger.info('Model loaded successfully')
    
    # 创建数据集和数据加载器
    logger.info('Creating test dataset...')
    test_dataset = PreferenceDataset(test_df, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    logger.info(f'Test batches: {len(test_loader)}')
    
    # 预测
    logger.info('Generating predictions...')
    predictions = predict(model, test_loader, device)
    logger.info('Predictions generated')
    
    # 创建提交文件
    logger.info('Creating submission file...')
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': predictions[:, 0],
        'winner_model_b': predictions[:, 1],
        'winner_tie': predictions[:, 2]
    })
    
    # 确保概率和为1（处理数值误差）
    prob_sum = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
    submission[['winner_model_a', 'winner_model_b', 'winner_tie']] = \
        submission[['winner_model_a', 'winner_model_b', 'winner_tie']].div(prob_sum, axis=0)
    
    # 保存
    submission.to_csv(output_file, index=False)
    logger.info(f'Submission saved to {output_file}')
    
    # 显示统计信息
    logger.info("\n" + "=" * 60)
    logger.info("Prediction Statistics:")
    logger.info(f"  Model A wins (avg): {submission['winner_model_a'].mean():.4f}")
    logger.info(f"  Model B wins (avg): {submission['winner_model_b'].mean():.4f}")
    logger.info(f"  Ties (avg): {submission['winner_tie'].mean():.4f}")
    logger.info("=" * 60)
    
    # 显示前几行
    logger.info("\nFirst 5 predictions:")
    logger.info("\n" + submission.head().to_string())
    
    logger.info("\n" + "=" * 60)
    logger.info("Inference completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
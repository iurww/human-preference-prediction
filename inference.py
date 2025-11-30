import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.logging_config import make_log_dir, init_logger
from configs import CONFIG, print_config
from dataset import HumanPreferenceTestDataset


def aggregate_multi_turn_predictions(predictions_dict, strategy='mean'):
    """
    聚合多轮对话的预测概率
    
    Args:
        predictions_dict: {original_id: [(id, probs), ...]}
        strategy: 聚合策略
            - 'mean': 平均概率（默认）
            - 'max': 取最大概率的那轮
            - 'vote': 投票机制（每轮选最可能的类别，最后统计）
            - 'weighted_mean': 加权平均（越靠后的轮次权重越大）
    
    Returns:
        {original_id: [prob_a, prob_b, prob_tie]}
    """
    aggregated = {}
    
    for original_id, turns in predictions_dict.items():
        if len(turns) == 1:
            aggregated[original_id] = turns[0][1]
        else:
            probs_array = np.array([prob for _, prob in turns])  # shape: (num_turns, 3)
            
            if strategy == 'mean':
                final_probs = probs_array.mean(axis=0)
                
            elif strategy == 'max':
                max_confidence = probs_array.max(axis=1)
                best_turn_idx = max_confidence.argmax()
                final_probs = probs_array[best_turn_idx]
                
            elif strategy == 'vote':
                votes = probs_array.argmax(axis=1)  # 每轮的预测类别
                vote_counts = np.bincount(votes, minlength=3)
                final_probs = vote_counts / vote_counts.sum()
                
            elif strategy == 'weighted_mean':
                weights = np.linspace(1, 2, len(turns))
                weights = weights / weights.sum()
                final_probs = (probs_array * weights[:, np.newaxis]).sum(axis=0)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            final_probs = final_probs / final_probs.sum()
            aggregated[original_id] = final_probs
    
    return aggregated


@torch.no_grad()
def inference(model, dataloader, device):

    model.eval()
    
    predictions = []
    cnt = 0
    progress_bar = tqdm(dataloader, desc='Inference')
    for batch in progress_bar:
        cnt += 1
        if cnt > 100:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_ids = batch['id']
        

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        for sample_id, prob in zip(batch_ids, probs):
            predictions.append((sample_id, prob))
    
    return predictions


def main():
    init_logger(make_log_dir())
    print_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    if torch.cuda.is_available():
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
    
    model_path = CONFIG.get('inference_model_path', f"{CONFIG['checkpoint_dir']}/best_model")
    logging.info(f'Loading model from: {model_path}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    
    logging.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    test_path = CONFIG.get('test_dataset_path', 'data/test.csv')
    test_df = pd.read_csv(test_path)
    logging.info(f'Loading test data from: {test_path}, Test samples: {len(test_df)}')
    
    test_dataset = HumanPreferenceTestDataset(
        data=test_df,
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        prompt_ratio=CONFIG.get('prompt_ratio', 0.3),
        cache_dir="./data",
        force_reprocess=False,
        usage="test"
    )
    
    logging.info(f'Processed test samples: {len(test_dataset)}')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logging.info('Starting inference...')
    predictions = inference(
        model, 
        test_loader, 
        device,
    )
    
    
    logging.info('Aggregating multi-turn predictions...')
    predictions_by_id = defaultdict(list)
    for sample_id, probs in predictions:
        # sample_id格式: "original_id_turn_index"
        original_id = '_'.join(sample_id.split('_')[:-1])  # 去掉最后的turn_index
        predictions_by_id[original_id].append((sample_id, probs))
    
    # 聚合多轮对话的预测
    aggregation_strategy = CONFIG.get('aggregation_strategy', 'mean')
    logging.info(f'Using aggregation strategy: {aggregation_strategy}')

    final_predictions = aggregate_multi_turn_predictions(
        predictions_by_id,
        strategy=aggregation_strategy
    )
    
    logging.info('Preparing submission file...')
    results = []
    for original_id in test_df['id']:
        original_id_str = str(original_id)
        if original_id_str in final_predictions:
            probs = final_predictions[original_id_str]
        else:
            logging.warning(f'ID {original_id} not found in predictions, using uniform distribution')
            probs = np.array([1/3, 1/3, 1/3])
        
        results.append({
            'id': original_id,
            'winner_model_a': f"{probs[0]:.6f}",
            'winner_model_b': f"{probs[1]:.6f}",
            'winner_tie': f"{probs[2]:.6f}"
        })
    
    output_path = CONFIG.get('submission_path', 'submission.csv')
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    
    logging.info(f'Submission saved to: {output_path}')
    logging.info(f'Total predictions: {len(submission_df)}')
    
    logging.info('\nSample predictions:')
    print(submission_df.head(10))
    
    logging.info('\nPrediction statistics:')
    logging.info(f"Average prob(model_a): {submission_df['winner_model_a'].astype(float).mean():.4f}")
    logging.info(f"Average prob(model_b): {submission_df['winner_model_b'].astype(float).mean():.4f}")
    logging.info(f"Average prob(tie): {submission_df['winner_tie'].astype(float).mean():.4f}")
    
    logging.info('\nInference completed!')


if __name__ == '__main__':
    main()
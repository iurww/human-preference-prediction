import os
import pandas as pd
import numpy as np
import logging
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.logging_config import make_log_dir, init_logger
from configs import CONFIG, print_config
from dataset import HumanPreferenceTestDataset


def aggregate_multi_turn_predictions(predictions_df, strategy='mean'):

    aggregated_results = []
    
    for original_id, group in predictions_df.groupby('original_id'):
        if len(group) == 1:
            row = group.iloc[0]
            aggregated_results.append({
                'id': int(original_id),
                'winner_model_a': row['winner_model_a'],
                'winner_model_b': row['winner_model_b'],
                'winner_tie': row['winner_tie']
            })
        else:
            probs_array = group[['winner_model_a', 'winner_model_b', 'winner_tie']].values  # shape: (num_turns, 3)
            
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
                weights = np.linspace(1, 2, len(group))
                weights = weights / weights.sum()
                final_probs = (probs_array * weights[:, np.newaxis]).sum(axis=0)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            final_probs = final_probs / final_probs.sum()
            
            aggregated_results.append({
                'id': int(original_id),
                'winner_model_a': final_probs[0],
                'winner_model_b': final_probs[1],
                'winner_tie': final_probs[2]
            })
    
    aggregated_df = pd.DataFrame(aggregated_results)
    
    return aggregated_df


@torch.no_grad()
def inference(model, dataloader, device):

    model.eval()
    
    results = []
    
    progress_bar = tqdm(dataloader, desc='Inference')
    for idx, batch in enumerate(progress_bar):
        
        if idx == 50:
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
            parts = sample_id.split('_')
            turn_index = parts[-1]
            original_id = '_'.join(parts[:-1])
            
            results.append({
                'id': sample_id,
                'original_id': original_id,
                'turn_index': turn_index,
                'winner_model_a': prob[0],
                'winner_model_b': prob[1],
                'winner_tie': prob[2]
            })
    
    predictions_df = pd.DataFrame(results)
    
    return predictions_df


def main():
    
    init_logger(make_log_dir())
    
    wandb.init(
        project=CONFIG.get('wandb_project', 'human-preference'),
        name=CONFIG.get('wandb_run_name', 'inference')
    )

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
    
    # 获取单论对话预测结果
    logging.info('Starting inference...')
    predictions_df = inference(model, test_loader, device)
    logging.info(f'Generated predictions for {len(predictions_df)} samples')
    
    raw_predictions_path = f"{CONFIG['checkpoint_dir']}/best_model/predictions_raw.csv"
    predictions_df.to_csv(raw_predictions_path, index=False)
    logging.info(f'Raw predictions saved to: {raw_predictions_path}')

    table = wandb.Table(dataframe=predictions_df)
    wandb.log({"raw_results": table})
    wandb.save(raw_predictions_path)
    logging.info(f'Raw predictions uploaded to wandb')
        
    
    # 多轮对话结果聚合
    aggregation_strategy = CONFIG.get('aggregation_strategy', 'mean')
    logging.info(f'Using aggregation strategy: {aggregation_strategy}')
    aggregated_df = aggregate_multi_turn_predictions(predictions_df, strategy=aggregation_strategy)

    agg_filled = test_df[['id']].drop_duplicates().merge(aggregated_df, on='id', how='left')
    fill_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    agg_filled[fill_cols] = agg_filled[fill_cols].fillna(1/3)

    aggregated_path = f"{CONFIG['checkpoint_dir']}/best_model/predictions_aggregated.csv"
    agg_filled.to_csv(aggregated_path, index=False)
    logging.info(f'Aggregated predictions saved to: {aggregated_path}')

    table = wandb.Table(dataframe=agg_filled)
    wandb.log({"aggregated_results": table})
    wandb.save(aggregated_path)
    logging.info(f'Aggregated predictions uploaded to wandb')
    

    wandb.finish()
    
if __name__ == '__main__':
    main()
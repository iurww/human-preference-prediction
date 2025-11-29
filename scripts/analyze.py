import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_json_field(field):
    """解析 JSON 格式的字段"""
    try:
        parsed = json.loads(field)
        if isinstance(parsed, list):
            return ' '.join(str(x) for x in parsed)
        return str(parsed)
    except:
        return str(field)

def get_token_length(row, tokenizer):
    """计算单个样本的token长度"""
    prompt = parse_json_field(row['prompt'])
    response_a = parse_json_field(row['response_a'])
    response_b = parse_json_field(row['response_b'])
    
    # 构建完整文本
    text = f"Prompt: {prompt}\n\nResponse A: {response_a}\n\nResponse B: {response_b}"
    
    # Tokenize
    tokens = tokenizer(text, add_special_tokens=True)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    response_a_tokens = tokenizer(response_a, add_special_tokens=False)
    response_b_tokens = tokenizer(response_b, add_special_tokens=False)
    
    return len(tokens['input_ids']), len(prompt_tokens['input_ids']), len(response_a_tokens['input_ids']), len(response_b_tokens['input_ids'])

def filter_long_samples(input_file, output_file, model_path, max_length=1024):
    """
    筛选出tokenize后超过max_length的样本
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        model_path: tokenizer模型路径
        max_length: 长度阈值
    """
    print(f"📂 加载数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ 总样本数: {len(df)}")
    
    print(f"\n🔧 加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"\n⏳ 开始筛选超过 {max_length} tokens 的样本...")
    
    long_samples = []
    token_lengths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        length, prompt_length, response_a_length, response_b_length = get_token_length(row, tokenizer)
        token_lengths.append(length)
        
        if length > max_length:
            # 添加长度信息列
            row_dict = row.to_dict()
            row_dict['token_length'] = length
            row_dict['prompt_length'] = prompt_length
            row_dict['response_a_length'] = response_a_length
            row_dict['response_b_length'] = response_b_length
            long_samples.append(row_dict)
    
    # 创建新的DataFrame
    long_df = pd.DataFrame(long_samples)
    
    # 按长度降序排序(最长的在最前面)
    long_df = long_df.sort_values('token_length', ascending=False)
    
    # 保存到CSV
    long_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 统计信息
    print(f"\n📊 统计结果:")
    print(f"  总样本数: {len(df)}")
    print(f"  超过 {max_length} 的样本数: {len(long_samples)}")
    print(f"  占比: {len(long_samples)/len(df)*100:.2f}%")
    print(f"  最长样本: {max(token_lengths)} tokens")
    print(f"  最短超长样本: {long_df['token_length'].min()} tokens" if len(long_samples) > 0 else "")
    print(f"\n✅ 已保存到: {output_file}")
    print(f"   (按token长度降序排列,最长的在最前面)")
    
    # 额外保存一个简化版(只包含关键信息,方便查看)
    if len(long_samples) > 0:
        simplified_output = output_file.replace('.csv', '_simplified.csv')
        simplified_df = long_df[['id', 'token_length', 'prompt', 'response_a', 'response_b', 
                                  'prompt_length', 'response_a_length', 'response_b_length',]]
        
        # 截断文本预览(只显示前100字符)
        for col in ['prompt', 'response_a', 'response_b']:
            simplified_df[col + '_preview'] = simplified_df[col].apply(
                lambda x: str(x)[:100] + '...' if len(str(x)) > 100 else str(x)
            )
            simplified_df = simplified_df.drop(columns=[col])
        
        simplified_df.to_csv(simplified_output, index=False, encoding='utf-8')
        print(f"   (简化版已保存到: {simplified_output})")

if __name__ == '__main__':
    # 配置参数
    INPUT_FILE = 'data/train_new.csv'
    OUTPUT_FILE = 'data/train_long_samples.csv'
    MODEL_PATH = './models/deberta'
    MAX_LENGTH = 1024
    
    filter_long_samples(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        model_path=MODEL_PATH,
        max_length=MAX_LENGTH
    )
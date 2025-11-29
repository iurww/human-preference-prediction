import pandas as pd
from tqdm import tqdm
import re

# 1. 白名单：允许英文字母、数字、常见可见符号及换行符
SAFE_PATTERN = re.compile(r'[A-Za-z0-9 !"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~\n]')

csv_path = 'data/train_new.csv'
df = pd.read_csv(csv_path)

def find_dirty_rows_with_chars(frame: pd.DataFrame):
    """
    返回 dict：{行号: {非法字符集合}}
    """
    dirty_info = {}
    for idx, row in tqdm(frame.astype(str).iterrows(), total=len(frame), desc='检查非法字符'):
        # 把整行所有字段拼成一个长字符串
        line = ''.join(row)
        # 把所有“合法”字符先全部替掉，剩下的就是非法字符
        bad = set(re.sub(SAFE_PATTERN, '', line))
        if bad:                      # 若集合非空，说明有非法字符
            dirty_info[idx] = bad
        # break
    return dirty_info

dirty_dict = find_dirty_rows_with_chars(df)

if dirty_dict:
    print('发现非法字符：')
    for row_idx, chars in dirty_dict.items():
        print(f'行号 {row_idx} -> {sorted(chars)}')
else:
    print('全部合法。')
    
print(len(dirty_dict))
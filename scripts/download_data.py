import os
import subprocess
import zipfile
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_extract():
    """下载并解压数据集"""
    
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    logger.info("Data directory created/verified")
    
    logger.info('Downloading dataset from Kaggle...')
    
    try:
        # 下载数据集
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', 'human-preference',
            '-p', 'data'
        ], check=True)
        logger.info('Dataset downloaded successfully')
    except subprocess.CalledProcessError as e:
        logger.error(f'Failed to download dataset: {e}')
        logger.error('Please check your Kaggle API credentials in ~/.kaggle/kaggle.json')
        return
    except FileNotFoundError:
        logger.error('Kaggle CLI not found. Please install it with: pip install kaggle')
        return
    
    logger.info('Extracting dataset...')
    
    # 解压文件
    zip_path = 'data/human-preference.zip'
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('data')
            logger.info('Dataset extracted successfully')
            
            # 删除压缩包
            os.remove(zip_path)
            logger.info('Zip file removed')
        except Exception as e:
            logger.error(f'Failed to extract dataset: {e}')
            return
    else:
        logger.error('Zip file not found. Please check the download.')
        return
    
    # 列出数据文件
    logger.info('\n' + '='*60)
    logger.info('Data files:')
    for file in os.listdir('data'):
        file_path = os.path.join('data', file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f'  - {file} ({size_mb:.2f} MB)')
    logger.info('='*60)


if __name__ == '__main__':
    download_and_extract()
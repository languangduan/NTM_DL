import argparse
from datetime import datetime
import logging
import os
import nltk


def setup_nltk_data(resources, base_path='nltk_data'):
    """
    设置NLTK数据目录并下载所需资源。

    参数:
    - resources: 需要下载的资源列表。
    - base_path: 存储NLTK数据的基本路径。
    """
    current_project_path = os.getcwd()
    nltk_data_path = os.path.join(current_project_path, base_path)

    # 创建目录以存储nltk数据
    os.makedirs(nltk_data_path, exist_ok=True)

    # 将自定义路径添加到nltk数据路径
    nltk.data.path.append(nltk_data_path)

    # 检查并下载每个资源
    for resource_name in resources:
        resource_path = os.path.join(nltk_data_path, resource_name)
        # 检查文件是否已存在
        if not os.path.exists(resource_path):
            print(f"Downloading {resource_name}...")
            nltk.download(resource_name, download_dir=nltk_data_path)
        else:
            print(f"{resource_name} already exists.")


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Topic Model Training')

    # 数据集相关参数
    parser.add_argument('--dataset', type=str, choices=['20news', 'movie'], default='movie',
                        help='Dataset to use (20news or movie)')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Categories for 20 NewsGroups dataset (optional)')
    parser.add_argument('--ngram_type', type=str, choices=['unigram', 'bigram'], default='bigram',
                        help='Type of n-gram to use')

    # 模型相关参数
    parser.add_argument('--supervised', type=bool, default=True,
                        help='Whether to use supervised learning')
    parser.add_argument('--regression', type=bool, default=True,
                        help='Whether to do regression')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of word embeddings')
    parser.add_argument('--topic_dim', type=int, default=100,
                        help='Dimension of topic embeddings')

    # 训练相关参数
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margin for ranking loss')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd',
                        help='Optimizer to use')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_dir', type=str, default='/scratch/yiyang/scratch/yiyang/group proj/datasets/rt_polaritydata/rt_polaritydata',
                        help='Data directory')

    return parser.parse_args()


def setup_logger(log_dir='logs'):
    """设置日志记录器"""
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # 配置日志记录器
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
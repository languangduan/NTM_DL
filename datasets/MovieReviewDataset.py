import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
import gensim.downloader as api
import nltk
import random

class PolarityDataset(Dataset):
    def __init__(self, data_dir, ngram_type='unigram', mode='supervised',  split='train',
                 test_size=0.2, num_negative_samples=1):
        self.data_dir = data_dir
        self.ngram_type = ngram_type
        self.vector_size = 300
        self.num_negative_samples = num_negative_samples
        self.model = None
        self.documents = []  # 存储所有文档及其标签
        self.data_pairs = []  # 存储最终的训练对
        self.mode = mode
        self.split = split
        self.test_size = test_size
        self.prepare_data()

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha()]
        return tokens

    def load_word2vec(self):
        print("Loading Word2Vec model...")
        self.model = api.load('word2vec-google-news-300')

    def ngram_to_embedding(self, ngram):
        embedding = np.zeros(self.vector_size)
        count = 0
        for token in ngram:
            if token in self.model:
                embedding += self.model[token]
                count += 1
        return embedding / count if count > 0 else embedding

    def document_to_embedding(self, tokens):
        embedding = np.zeros(self.vector_size)
        count = 0
        for token in tokens:
            if token in self.model:
                embedding += self.model[token]
                count += 1
        return embedding / count if count > 0 else embedding

    def generate_ngrams(self, tokens):
        if self.ngram_type == 'unigram':
            return [(token,) for token in tokens]
        elif self.ngram_type == 'bigram':
            return [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        else:
            raise ValueError("Unsupported ngram_type. Choose 'unigram' or 'bigram'.")

    def read_data(self):
        # 读取负面评论
        with open(os.path.join(self.data_dir, 'rt-polarity.neg'), 'r', encoding='latin-1') as f:
            neg_lines = f.readlines()
        # 读取正面评论
        with open(os.path.join(self.data_dir, 'rt-polarity.pos'), 'r', encoding='latin-1') as f:
            pos_lines = f.readlines()

        all_documents = []
        # 处理所有文档
        for line in neg_lines:
            tokens = self.preprocess(line.strip())
            if tokens:  # 确保文档非空
                doc_embedding = self.document_to_embedding(tokens)
                ngrams = self.generate_ngrams(tokens)
                all_documents.append({
                    'embedding': doc_embedding,
                    'ngrams': [self.ngram_to_embedding(ngram) for ngram in ngrams],
                    'polarity': 0  # 负面评论
                })

        for line in pos_lines:
            tokens = self.preprocess(line.strip())
            if tokens:  # 确保文档非空
                doc_embedding = self.document_to_embedding(tokens)
                ngrams = self.generate_ngrams(tokens)
                all_documents.append({
                    'embedding': doc_embedding,
                    'ngrams': [self.ngram_to_embedding(ngram) for ngram in ngrams],
                    'polarity': 1  # 正面评论
                })
        # 划分训练集和测试集
        train_docs, test_docs = train_test_split(
            all_documents,
            test_size=self.test_size,
            random_state=42,
            shuffle=True
        )

        # 根据split参数选择相应的数据集
        self.documents = train_docs if self.split == 'train' else test_docs
        print(f"{self.split} set size: {len(self.documents)} documents")

    def generate_training_pairs(self):
        if self.mode == 'unsupervised':
            # 无监督模式：每个样本包含(doc, positive_ngram, negative_ngram, polarity)
            for i, doc in enumerate(self.documents):
                doc_embedding = doc['embedding']
                doc_polarity = doc['polarity']

                # 对每个n-gram生成训练对
                for pos_ngram in doc['ngrams']:
                    # 随机选择一个不同的文档作为负例
                    while True:
                        neg_doc_idx = random.randint(0, len(self.documents) - 1)
                        if neg_doc_idx != i:
                            break

                    neg_doc = self.documents[neg_doc_idx]
                    neg_ngram = random.choice(neg_doc['ngrams'])

                    self.data_pairs.append({
                        'doc_embedding': doc_embedding,
                        'pos_ngram_embedding': pos_ngram,
                        'neg_ngram_embedding': neg_ngram,
                        'doc_polarity': doc_polarity
                    })

        else:  # supervised mode
            # 监督模式：每个样本包含(doc, ngram, label, polarity)
            for i, doc in enumerate(self.documents):
                doc_embedding = doc['embedding']
                doc_polarity = doc['polarity']

                # 添加正例
                for ngram in doc['ngrams']:
                    self.data_pairs.append({
                        'doc_embedding': doc_embedding,
                        'ngram_embedding': ngram,
                        'pair_label': 1,
                        'doc_polarity': doc_polarity
                    })

                # 添加负例
                for _ in range(len(doc['ngrams']) * self.num_negative_samples):
                    # 随机选择一个不同的文档
                    while True:
                        neg_doc_idx = random.randint(0, len(self.documents) - 1)
                        if neg_doc_idx != i:
                            break

                    neg_doc = self.documents[neg_doc_idx]
                    neg_ngram = random.choice(neg_doc['ngrams'])

                    self.data_pairs.append({
                        'doc_embedding': doc_embedding,
                        'ngram_embedding': neg_ngram,
                        'pair_label': 0,
                        'doc_polarity': doc_polarity
                    })


    def prepare_data(self):
        self.load_word2vec()
        self.read_data()
        self.generate_training_pairs()
        # 清理中间数据节省内存
        self.documents = None

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        if self.mode == 'unsupervised':
            return (
                torch.tensor(item['doc_embedding'], dtype=torch.float32),
                torch.tensor(item['pos_ngram_embedding'], dtype=torch.float32),
                torch.tensor(item['neg_ngram_embedding'], dtype=torch.float32),
                torch.tensor(item['doc_polarity'], dtype=torch.long)
            )
        else:  # supervised mode
            return (
                torch.tensor(item['doc_embedding'], dtype=torch.float32),
                torch.tensor(item['ngram_embedding'], dtype=torch.float32),
                torch.tensor(item['pair_label'], dtype=torch.float32),
                torch.tensor(item['doc_polarity'], dtype=torch.long)
            )
if __name__ == '__main__':
    # 下载必要的NLTK数据
    nltk.download('punkt')

    # 创建数据集实例
    data_dir = 'path/to/your/rt-polaritydata'
    dataset = PolarityDataset(
        data_dir=data_dir,
        ngram_type='unigram',
        num_negative_samples=1  # 每个正样本生成的负样本数量
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # 测试数据加载
    for batch_idx, (doc_emb, ngram_emb, pair_label, doc_polarity) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Document embeddings shape: {doc_emb.shape}")
        print(f"N-gram embeddings shape: {ngram_emb.shape}")
        print(f"Pair labels shape: {pair_label.shape}")
        print(f"Document polarity labels shape: {doc_polarity.shape}")
        break

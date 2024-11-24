import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
import gensim.downloader as api
from nltk import word_tokenize
import nltk
import random

from utils.utils import setup_nltk_data


class NewsGroupDataset(Dataset):
    def __init__(self, categories=None, ngram_type='unigram', split='train', mode='supervised'):
        self.categories = categories
        self.ngram_type = ngram_type
        self.vector_size = 300  # Assuming 300-dimensional vectors
        self.model = None
        self.data_pairs = []
        self.target_labels = []
        self.mode = mode
        self.split = split
        self.prepare_data()

    def load_data(self):
        if self.split == 'train':
            data_train = fetch_20newsgroups(subset='train', categories=self.categories, shuffle=True, random_state=42)
        else:
            data_train = fetch_20newsgroups(subset='test', categories=self.categories, shuffle=True, random_state=42)
        return data_train

    def preprocess(self, text):
        # 分词
        tokens = word_tokenize(text.lower())
        # 去除非字母字符
        tokens = [word for word in tokens if word.isalpha()]
        return tokens

    def load_word2vec(self):
        # 使用gensim的api下载并加载预训练的Word2Vec模型
        print("Loading Word2Vec model...")
        self.model = api.load('word2vec-google-news-300')

    def ngram_to_embedding(self, ngram):
        # 将n-gram转换为嵌入表示
        embedding = np.zeros(self.vector_size)
        for token in ngram:
            if token in self.model:
                embedding += self.model[token]
        return embedding / len(ngram) if ngram else embedding

    def document_to_embedding(self, doc):
        tokens = self.preprocess(doc)
        embedding = np.zeros(self.vector_size)
        for token in tokens:
            if token in self.model:
                embedding += self.model[token]
        return embedding / len(tokens) if tokens else embedding

    def generate_ngrams(self, tokens):
        if self.ngram_type == 'unigram':
            return [(token,) for token in tokens]
        elif self.ngram_type == 'bigram':
            return [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        else:
            raise ValueError("Unsupported ngram_type. Choose 'unigram' or 'bigram'.")

    def prepare_data(self):
        self.load_word2vec()
        data_train = self.load_data()
        documents = [self.preprocess(doc) for doc in data_train.data]
        self.target_labels = data_train.target
        if self.mode == 'unsupervised':
            for doc_tokens, target in zip(documents, self.target_labels):
                doc_embedding = self.document_to_embedding(' '.join(doc_tokens))
                ngrams = self.generate_ngrams(doc_tokens)

                # 生成正负样本对
                for ngram in ngrams:
                    ngram_embedding_pos = self.ngram_to_embedding(ngram)
                    while True:
                        random_doc = random.choice(documents)  # 假设 docs 是文档列表
                        if random_doc != doc_tokens:
                            break
                    random_ngram = self.generate_ngrams(random_doc)[0]
                    ngram_embedding_neg = self.ngram_to_embedding(random_ngram)
                    self.data_pairs.append((doc_embedding, ngram_embedding_pos, ngram_embedding_neg, target))
        else:
            for doc_tokens, target in zip(documents, self.target_labels):
                doc_embedding = self.document_to_embedding(' '.join(doc_tokens))
                ngrams = self.generate_ngrams(doc_tokens)

                # 生成正样本
                for ngram in ngrams:
                    ngram_embedding = self.ngram_to_embedding(ngram)
                    self.data_pairs.append((doc_embedding, ngram_embedding, 1, target))

                # 生成负样本
                for _ in range(len(ngrams)):
                    random_doc = random.choice(documents)
                    if random_doc != doc_tokens:
                        random_ngram = self.generate_ngrams(random_doc)[0]
                        ngram_embedding = self.ngram_to_embedding(random_ngram)
                        self.data_pairs.append((doc_embedding, ngram_embedding, 0, target))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if self.mode == 'unsupervised':
            doc_embedding, ngram_embedding_pos, ngram_embedding_neg, target_label = self.data_pairs[idx]
            return (
                torch.tensor(doc_embedding, dtype=torch.float32),
                torch.tensor(ngram_embedding_pos, dtype=torch.float32),
                torch.tensor(ngram_embedding_neg, dtype=torch.float32),
                torch.tensor(target_label, dtype=torch.long)
            )
        else:
            doc_embedding, ngram_embedding, label, target_label = self.data_pairs[idx]
            return (
                torch.tensor(doc_embedding, dtype=torch.float32),
                torch.tensor(ngram_embedding, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(target_label, dtype=torch.long)
            )

if __name__ == '__main__':
    # 准备punkt资源
    required_resources = ['punkt', 'punkt_tab']
    setup_nltk_data(required_resources)

    # 创建数据集实例，选择unigram或bigram
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    dataset = NewsGroupDataset(categories=categories, ngram_type='bigram')  # 选择'unigram'或'bigram'

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, (doc_emb, ngram_emb, label, target_label) in enumerate(dataloader):
        print("Batch index:", batch_idx)
        print("Document Embedding shape:", doc_emb.shape)
        print("N-gram Embedding shape:", ngram_emb.shape)
        print("Label shape:", label.shape)
        print("Target Label shape:", target_label.shape)
        break  # 只检查第一个批次

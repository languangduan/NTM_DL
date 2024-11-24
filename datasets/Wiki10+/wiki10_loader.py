import json
import torch
from torch.utils.data import Dataset, DataLoader
from document_read import document_read
import gensim.downloader as api
from nltk import word_tokenize
import numpy as np
import random
import h5py

# 自定义数据集类
class wiki10Dataset(Dataset):
    def __init__(self, file_path):
        # 读取  文件
        self.file_path=file_path
        with h5py.File(self.file_path, 'r') as f:
            self.doc_embedding = f['doc_embedding']
            self.length=len(self.doc_embedding)


    def __len__(self):
        # 返回数据长度
        #print(self.length)
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            self.doc_embedding = f['doc_embedding']
            self.ngram_embedding=f['ngram_embedding']
            self.label=f['label']
            self.target=f['target']
            doc_embedding=self.doc_embedding[idx]
            ngram_embedding=self.ngram_embedding[idx]
            label=self.label[idx]
            target_label =self.target[idx]

        return (
            torch.tensor(doc_embedding, dtype=torch.float32),
            torch.tensor(ngram_embedding, dtype=torch.float32),
            torch.tensor(label[0], dtype=torch.float32),
            torch.tensor(target_label, dtype=torch.float32)
        )



if __name__ == "__main__":
    dataset = wiki10Dataset('unigram_test_dataset.h5')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for doc_embedding, ngram_embedding, label, target_label in data_loader:
        print(doc_embedding.shape)
        #print(doc_embedding)
        print(ngram_embedding.shape)
        #print(ngram_embedding)
        print(label.shape)
        #print(label)
        print(target_label.shape)
        #print(target_label)
        break

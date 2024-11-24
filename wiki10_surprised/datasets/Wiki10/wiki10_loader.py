import json
import torch
from torch.utils.data import Dataset, DataLoader
from document_read2 import document_read
import gensim.downloader as api
from nltk import word_tokenize
import numpy as np
import random
import h5py

# 自定义数据集类

class Wiki10Dataset(Dataset):
    def __init__(self,json_path,document_base_path, ngram_type='unigram', mode='supervised'):
        self.ngram_type = ngram_type
        self.vector_size = 300  # Assuming 300-dimensional vectors
        self.model = None
        self.data_pairs = []
        self.target_labels = []
        self.mode = mode
        self.json_path=json_path
        self.document_base_path=document_base_path
        self.prepare_data()
    
    def generate_multilabel_vector(self,vec1,vec2):
        '''
        this function will compare all the elements in vec2, if it exist in vec1, the the posityion is 1,else it's 0
        vec1 is the categories of the document, vec2 is all possible categories.
        '''
        set_vector1 = set(vec1)
        result=[1 if element in set_vector1 else 0 for element in vec2]
        return np.array(result)

    def load_data(self):
        with open(self.json_path, 'r') as file:
            self.data = json.load(file)
            self.categories=self.data['categories']
            self.data=self.data['hash_categories_pair']   #  alist with every element as ['hash',tltle,[cate1,cate2...]]
        mydocuments=[]
        mylabel_vecs=[]
        for doc_info in self.data:
            document=self.preprocess(document_read(self.document_base_path+'/'+doc_info[0]))
            label_vec=self.generate_multilabel_vector(doc_info[2],self.categories)
            if len(document)>0 :
                mydocuments.append(document)
                mylabel_vecs.append(label_vec)

        return  mydocuments,mylabel_vecs

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
        documents,self.target_labels =  self.load_data()

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
                torch.tensor(target_label, dtype=torch.float32)
            )
        else:
            doc_embedding, ngram_embedding, label, target_label = self.data_pairs[idx]
            return (
                torch.tensor(doc_embedding, dtype=torch.float32),
                torch.tensor(ngram_embedding, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(target_label, dtype=torch.float32)
            )


if __name__ == "__main__":
    dataset = Wiki10Dataset('new_dataset_for_25_top_cat_test.json','wiki10+_documents/documents')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for doc_embedding, ngram_embedding, label, target_label in data_loader:
        print(doc_embedding.shape)
        print(doc_embedding)
        print(ngram_embedding.shape)
        print(ngram_embedding)
        print(label.shape)
        print(label)
        print(target_label.shape)
        print(target_label)
        break

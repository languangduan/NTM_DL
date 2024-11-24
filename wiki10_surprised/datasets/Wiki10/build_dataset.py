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
class build_dataset():
    def __init__(self, json_file,document_base_path,outfile_path,ngram_type='unigram', mode='supervised'):
        # 读取 JSON 文件
        with open(json_file, 'r') as file:
            self.data = json.load(file)
            self.ngram_type = ngram_type
            self.categories=self.data['categories']
            self.data=self.data['hash_categories_pair']   #  alist with every element as ['hash',tltle,[cate1,cate2...]]
        self.document_base_path=document_base_path
        self.model=None
        self.outfile_path=outfile_path
        self.vector_size = 300  # Assuming 300-dimensional vectors
        self.data_pairs = []
        self.target_labels = []
        self.ngram_type=ngram_type
        self.mode=mode

        self.prepare_data()

    def generate_multilabel_vector(self,vec1,vec2):
        '''
        this function will compare all the elements in vec2, if it exist in vec1, the the posityion is 1,else it's 0
        vec1 is the categories of the document, vec2 is all possible categories.
        '''
        set_vector1 = set(vec1)
        result=[1 if element in set_vector1 else 0 for element in vec2]
        return np.array(result)


    def prepare_data(self):
        '''
        this is used to generate a list for tonkes for documents and the lables(one vector can contain more than 1 target, as it's multi-label)
        '''
        self.load_word2vec()
        mydocuments=[]
        mylabel_vecs=[]
        #process the dataset one by one
        for doc_info in self.data:
            document=self.preprocess(document_read(self.document_base_path+'/'+doc_info[0]))
            label_vec=self.generate_multilabel_vector(doc_info[2],self.categories)
            if len(document)>0:
                mydocuments.append(document)
                mylabel_vecs.append(label_vec)
        #print(len(mydocuments))
        #print(len( mylabel_vecs))
        if self.mode == 'supervised':
            with h5py.File('supervised_'+self.ngram_type+'_'+self.outfile_path, "w") as f:
                dset_doc_embedding = f.create_dataset('doc_embedding', (0,self.vector_size), maxshape=(None,self.vector_size))
                dset_ngram_embedding = f.create_dataset('ngram_embedding', (0,self.vector_size), maxshape=(None,self.vector_size))
                dset_label = f.create_dataset('label', (0,1), maxshape=(None,1))
                dset_target = f.create_dataset('target', (0,len(self.categories)), maxshape=(None,len(self.categories)))

                for doc_tokens, target in zip(mydocuments, mylabel_vecs):
                    doc_embedding = self.document_to_embedding(' '.join(doc_tokens))
                    ngrams = self.generate_ngrams(doc_tokens)

                    # 生成正样本
                    for ngram in ngrams:
                        ngram_embedding = self.ngram_to_embedding(ngram)
                        #print([doc_embedding, ngram_embedding, 1, target])
                        dset_doc_embedding.resize(dset_doc_embedding.shape[0] + 1, axis=0) 
                        dset_ngram_embedding.resize(dset_ngram_embedding.shape[0] + 1, axis=0)
                        dset_label.resize(dset_label.shape[0] + 1, axis=0) 
                        dset_target.resize(dset_target.shape[0] + 1, axis=0) 
                        dset_doc_embedding[-1]=doc_embedding
                        dset_ngram_embedding[-1]=ngram_embedding
                        dset_label[-1]=1
                        dset_target[-1]=target



                    # 生成负样本
                    for _ in range(len(ngrams)):
                        random_doc = random.choice(mydocuments)
                        if random_doc != doc_tokens:
                            random_ngram = self.generate_ngrams(random_doc)[0]
                            ngram_embedding = self.ngram_to_embedding(random_ngram)
                            dset_doc_embedding.resize(dset_doc_embedding.shape[0] + 1, axis=0) 
                            dset_ngram_embedding.resize(dset_ngram_embedding.shape[0] + 1, axis=0)
                            dset_label.resize(dset_label.shape[0] + 1, axis=0) 
                            dset_target.resize(dset_target.shape[0] + 1, axis=0) 
                            dset_doc_embedding[-1]=doc_embedding
                            dset_ngram_embedding[-1]=ngram_embedding
                            dset_label[-1]=0
                            dset_target[-1]=target


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
        print('Finished loading')

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





if __name__ == "__main__":
    dataset = build_dataset('new_dataset_for_25_top_cat_test.json','wiki10+_documents/documents','test_dataset.h5')
    dataset = build_dataset('new_dataset_for_25_top_cat_train.json','wiki10+_documents/documents','train_dataset.h5')


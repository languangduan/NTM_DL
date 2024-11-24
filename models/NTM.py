import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralTopicModel(nn.Module):
    def __init__(self, embedding_dim=300, topic_dim=100):
        super(NeuralTopicModel, self).__init__()
        # n-gram embedding to topic layer
        self.ngram_to_topic = nn.Linear(embedding_dim, topic_dim)
        # document embedding to topic layer
        self.document_to_topic = nn.Linear(embedding_dim, topic_dim)

    def forward(self, ngram_embedding, document_embedding):
        # n-gram topic representation
        ngram_topic = torch.sigmoid(self.ngram_to_topic(ngram_embedding))
        # document topic representation
        document_topic = F.softmax(self.document_to_topic(document_embedding), dim=-1)
        # calculate matching score (dot product)
        score = torch.sum(ngram_topic * document_topic, dim=-1)
        return score

class SupervisedNeuralTopicModel(nn.Module):
    def __init__(self, embedding_dim=300, topic_dim=100, num_labels=10):
        super(SupervisedNeuralTopicModel, self).__init__()
        self.ngram_to_topic = nn.Linear(embedding_dim, topic_dim)
        self.document_to_topic = nn.Linear(embedding_dim, topic_dim)
        self.topic_to_label = nn.Linear(topic_dim, num_labels)

    def forward(self, ngram_embedding, document_embedding):
        ngram_topic = torch.sigmoid(self.ngram_to_topic(ngram_embedding))
        document_topic = F.softmax(self.document_to_topic(document_embedding), dim=-1)
        score = torch.sum(ngram_topic * document_topic, dim=-1)
        label_logits = self.topic_to_label(document_topic)
        label_probs = F.softmax(label_logits, dim=-1)
        return score, label_probs


class UnifiedNeuralTopicModel(nn.Module):
    def __init__(self, embedding_dim=300, topic_dim=100, num_labels=None):
        super(UnifiedNeuralTopicModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.topic_dim = topic_dim
        self.supervised_mode = num_labels is not None

        # 共享的基础层
        self.ngram_to_topic = nn.Linear(embedding_dim, topic_dim)
        self.document_to_topic = nn.Linear(embedding_dim, topic_dim)

        # 监督模式的额外层
        if self.supervised_mode:
            self.topic_to_label = nn.Linear(topic_dim, num_labels)

    def forward(self, doc_embedding, ngram_embedding):
        ngram_topic = torch.sigmoid(self.ngram_to_topic(ngram_embedding))
        document_topic = F.softmax(self.document_to_topic(doc_embedding), dim=-1)
        score = torch.sum(ngram_topic * document_topic, dim=-1)

        if self.supervised_mode:
            label_logits = self.topic_to_label(document_topic)
            label_probs = F.softmax(label_logits, dim=-1)
            return score, label_probs
        return score


if __name__ == '__main__':

    batch_size = 32
    embedding_dim = 300
    num_labels = 10

    ngram_embeddings = torch.randn(batch_size, embedding_dim)
    document_embeddings = torch.randn(batch_size, embedding_dim)

    model = SupervisedNeuralTopicModel(embedding_dim=embedding_dim, topic_dim=100, num_labels=num_labels)

    scores, label_logits = model(ngram_embeddings, document_embeddings)

    print("匹配概率:", scores)
    print("标签预测:", label_logits)

    # # 初始化模型
    # model = NeuralTopicModel(embedding_dim=embedding_dim, topic_dim=100)
    #
    # # 前向传播
    # scores = model(ngram_embeddings, document_embeddings)
    #
    # # 输出匹配概率
    # print("匹配概率:", scores)



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import f1_score
import sys
sys.path.append('datasets/Wiki10')
from datasets.Wiki10.wiki10_loader import Wiki10Dataset
from datasets.Wiki10.tag_extract import build_new_tag_for_25actegory
import numpy as np
import logging





class WikiSupervisedNeuralTopicModel(nn.Module):
    def __init__(self, embedding_dim=300, topic_dim=100, num_labels=10):
        super(WikiSupervisedNeuralTopicModel, self).__init__()
        self.ngram_to_topic = nn.Linear(embedding_dim, topic_dim)
        self.document_to_topic = nn.Linear(embedding_dim, topic_dim)
        self.topic_to_label = nn.Linear(topic_dim, num_labels)

    def forward(self, ngram_embedding, document_embedding):
        ngram_topic = torch.sigmoid(self.ngram_to_topic(ngram_embedding))
        document_topic = F.softmax(self.document_to_topic(document_embedding), dim=-1)
        score = torch.sum(ngram_topic * document_topic, dim=-1)
        label_logits = self.topic_to_label(document_topic)
        label_probs = F.sigmoid(label_logits)
        return score, label_probs





def train_model(model, train_dataset, val_dataset=None,
                num_epochs=10, batch_size=32, learning_rate=0.01,
                margin=0.5, pretrain=True, supervised=True,
                label_weight=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    def pretrain_step():
        """预训练步骤"""
        print("Starting pretraining...")
        pretraining_epoches=10
        pretrain_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(pretraining_epoches):
            model.train()
            total_loss = 0

            for batch in tqdm(train_dataloader):
                doc_emb, ngram_emb = batch[0].to(device), batch[1].to(device)

                # 自编码器重构损失
                hidden = torch.sigmoid(model.ngram_to_topic(ngram_emb))
                reconstructed = torch.sigmoid(model.ngram_to_topic.weight.t() @ hidden.t()).t()
                reconstruction_loss = F.mse_loss(reconstructed, ngram_emb)

                pretrain_optimizer.zero_grad()
                reconstruction_loss.backward()
                pretrain_optimizer.step()

                total_loss += reconstruction_loss.item()

            print(f"Pretrain Epoch {epoch + 1}/{pretraining_epoches}, Loss: {total_loss / len(train_dataloader):.4f}")
            logging.info(f"Pretrain Epoch {epoch + 1}/{pretraining_epoches}, Loss: {total_loss / len(train_dataloader):.4f}")
            torch.save(model,'pretrained_model.pth')

    def train_step():
        """主训练步骤"""
        print("Starting main training...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        classification_criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_dataloader):
                if train_dataset.mode == 'unsupervised':
                    # 解包无监督模式的数据
                    doc_emb, ngram_emb_pos, ngram_emb_neg, target = [x.to(device) for x in batch]

                    # 计算正例和负例的得分
                    if supervised:
                        pos_score, label_probs = model(doc_emb, ngram_emb_pos)
                        neg_score = model(doc_emb, ngram_emb_neg)[0]
                    else:
                        pos_score = model(doc_emb, ngram_emb_pos)
                        neg_score = model(doc_emb, ngram_emb_neg)

                    # 计算排序损失

                    ranking_loss = torch.clamp(margin - pos_score + neg_score, min=0).mean()

                else:
                    # 解包监督模式的数据
                    doc_emb, ngram_emb, label, target = [x.to(device) for x in batch]

                    # 计算得分和标签预测
                    if supervised:
                        score, label_probs = model(doc_emb, ngram_emb)
                        ranking_loss = F.binary_cross_entropy_with_logits(score, label)
                    else:
                        score = model(doc_emb, ngram_emb)
                        ranking_loss = F.binary_cross_entropy_with_logits(score, label)

                # 计算总损失
                if supervised:
                    label_loss = classification_criterion(label_probs, target)
                    loss = (1 - label_weight) * ranking_loss + label_weight * label_loss
                else:
                    loss = ranking_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")
            logging.info(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")
            torch.save(model, 'whole_model.pth')

            # 验证
            if val_dataset:
                validate(model, val_dataloader, supervised, device)

    # 执行训练流程
    if pretrain:
        pretrain_step()
    train_step()



def validate(model, val_dataloader, supervised, device,threshold=0.2):
    """验证函数，返回损失和各项指标"""
    model.eval()
    total_loss = 0
    corrected=0
    all_label_added=0
    all_predictions = []
    all_labels = []
    classification_criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in val_dataloader:
            pos_doc, pos_ngram = [x.to(device) for x in batch[:2]]

            if supervised and len(batch) > 3:
                labels = batch[3].to(device)
                scores, label_probs = model(pos_ngram, pos_doc)

                # 计算损失
                loss = classification_criterion(label_probs, labels)


                # 收集预测和真实标签
                all_predictions.extend(np.where(label_probs.cpu().numpy() >= threshold, 1, 0))
                all_labels.extend(labels.cpu().numpy())

                all_label_added+=np.sum(labels.cpu().numpy())
                corrected+=np.sum(np.multiply(labels.cpu().numpy(),np.where(label_probs.cpu().numpy() >= threshold, 1, 0)))

            else:
                # 非监督情况
                scores = model(pos_ngram, pos_doc)
                loss = -scores.mean()  # 简单的得分最大化

                # 对于非监督学习，使用阈值0.5
                predicted = (scores > 0.5).float()
                all_predictions.extend(predicted.cpu().numpy())
                if len(batch) > 3:  # 如果有标签
                    all_labels.extend(batch[3].numpy())

            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(val_dataloader)

    # 如果有标签数据，计算各项指标
    if all_labels:
        # 计算Micro-F1
        micro_f1 = f1_score(all_labels, all_predictions, average='micro',zero_division=1)

        # 计算Macro-F1
        macro_f1 = f1_score(all_labels, all_predictions, average='macro',zero_division=1)

        # 计算整体准确率
        if supervised:
            accuracy = corrected / all_label_added
        else:
            accuracy = sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    #print(avg_loss)
    #print(accuracy)
    #print(micro_f1)
    #print(macro_f1)
    print(f"Loss: {avg_loss:.4f},Accuracy: {accuracy:.4f},Micro-F1: {micro_f1:.4f},Macro-F1: {macro_f1:.4f}")
    logging.info(f"Loss: {avg_loss:.4f},Accuracy: {accuracy:.4f},Micro-F1: {micro_f1:.4f},Macro-F1: {macro_f1:.4f}")


if __name__ == '__main__':

    if True:
        build_new_tag_for_25actegory(doc_length=None,file_path='datasets/Wiki10/wiki10+_tag-data/tag-data.xml',train_out_filepath='datasets/Wiki10/new_dataset_for_25_top_cat_train.json',test_out_filepath='datasets/Wiki10/new_dataset_for_25_top_cat_test.json',choosen_cate_number=25,split_proportion=4)
        print('prepare data ok!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode='supervised'
   
    train_dataset = Wiki10Dataset(mode=mode, json_path='datasets/Wiki10/new_dataset_for_25_top_cat_train.json',document_base_path='datasets/Wiki10/wiki10+_documents/documents')
    test_dataset = Wiki10Dataset(mode=mode, json_path='datasets/Wiki10/new_dataset_for_25_top_cat_test.json',document_base_path='datasets/Wiki10/wiki10+_documents/documents')


    leng = len(train_dataset.categories)
    #print(leng)

    model = WikiSupervisedNeuralTopicModel(embedding_dim=300, topic_dim=100, num_labels=leng)


    #训练和测试，后续加上验证
    train_model(model, train_dataset, val_dataset=test_dataset,num_epochs=100,label_weight=0.8,supervised=(mode=='supervised'))

    # test_model(test_loader, model, device)

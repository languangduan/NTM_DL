import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, r2_score
from tqdm import tqdm

from datasets.TwentyNewsDataset import NewsGroupDataset
from datasets.MovieReviewDataset import PolarityDataset
from models.NTM import SupervisedNeuralTopicModel, NeuralTopicModel
from utils.utils import setup_nltk_data, parse_args, setup_logger


def train_model(model, train_dataset, val_dataset=None,
                num_epochs=10, batch_size=32, learning_rate=0.01,
                margin=0.5, pretrain=True, supervised=True,
                label_weight=0.5, regression_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    def pretrain_step():
        """预训练步骤"""
        logger.info("Starting pretraining...")
        pretrain_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs // 2):
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

            logger.info(f"Pretrain Epoch {epoch + 1}/{num_epochs // 2}, Loss: {total_loss / len(train_dataloader):.4f}")

    def train_step():
        """主训练步骤"""
        logger.info("Starting main training...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                    if not regression_mode:
                        label_loss = F.cross_entropy(label_probs, target)
                        loss = (1 - label_weight) * ranking_loss + label_weight * label_loss
                    else:
                        label_loss = F.mse_loss(label_probs, target.float())
                        loss = (1 - label_weight) * ranking_loss + label_weight * label_loss
                else:
                    loss = ranking_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

            # 验证
            if val_dataset:
                validate(model, val_dataloader, supervised, device, regression_mode)

    # 执行训练流程
    if pretrain:
        pretrain_step()
    train_step()


def validate(model, val_dataloader, supervised, device, regression_mode=False):
    """验证函数，返回损失和各项指标"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_scores = []  # 用于存储回归模式的原始预测分数

    with torch.no_grad():
        for batch in val_dataloader:
            pos_doc, pos_ngram = [x.to(device) for x in batch[:2]]

            if supervised and len(batch) > 3:
                labels = batch[3].to(device)
                if regression_mode:
                    scores = model(pos_ngram, pos_doc)[0]
                    loss = F.mse_loss(scores.squeeze(), labels.float())

                    # 存储原始预测分数
                    all_scores.extend(scores.squeeze().cpu().numpy())

                    # 使用0.5作为阈值进行二分类
                    predicted = (scores.squeeze() > 0.5).float()
                else:
                    scores, label_probs = model(pos_ngram, pos_doc)
                    loss = F.cross_entropy(label_probs, labels)
                    predicted = torch.argmax(label_probs, dim=1)

                # 收集预测和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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
        micro_f1 = f1_score(all_labels, all_predictions, average='micro')

        # 计算Macro-F1
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')

        # 计算整体准确率
        accuracy = sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)

        # 计算R²分数（如果需要的话）
        r2 = None
        if regression_mode:
            r2 = r2_score(all_labels, all_predictions)

        # 输出所有指标
        logger.info(f"Validation Results:")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Micro-F1: {micro_f1:.4f}")
        logger.info(f"Macro-F1: {macro_f1:.4f}")
        if r2 is not None:
            logger.info(f"R²: {r2:.4f}")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'r2': r2
        }
    else:
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        return {'loss': avg_loss}


def test_model(dataloader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    model.to(device)

    with torch.no_grad():
        for doc_emb, ngram_emb, _, target_label in dataloader:
            doc_emb = doc_emb.to(device)
            ngram_emb = ngram_emb.to(device)
            target_label = target_label.to(device)

            _, label_probs = model(ngram_emb, doc_emb)
            preds = torch.argmax(label_probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target_label.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f'Test Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger()

    # 记录配置信息
    logger.info("=== Training Configuration ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Supervised: {args.supervised}")
    logger.info(f"Number of Epochs: {args.num_epochs}")
    logger.info("===========================")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备punkt资源
    required_resources = ['punkt', 'punkt_tab']
    setup_nltk_data(required_resources)

    dataset = None
    dataset_val = None
    if args.supervised:
        mode = 'supervised'
    else:
        mode = 'unsupervised'
    # 准备数据集
    if args.dataset == '20news':
        dataset = NewsGroupDataset(categories=args.categories, ngram_type=args.ngram_type, split='train', mode=mode)
        dataset_val = NewsGroupDataset(categories=args.categories, ngram_type=args.ngram_type, split='test', mode=mode)
        num_labels = len(args.categories) if args.categories else 20
    elif args.dataset == 'movie':  # movie reviews
        dataset = PolarityDataset(data_dir=args.data_dir, split='train', mode=mode)
        dataset_val = PolarityDataset(data_dir=args.data_dir, split='test', mode=mode)
        if args.regression:
            num_labels = 1
        else:
            num_labels = 2

    train_dataset, val_dataset = train_test_split(
        dataset,
        test_size=0.2,  # 测试集占20%
        random_state=42,  # 设置随机种子，确保结果可复现
        shuffle=True  # 确保数据被打乱
    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    if args.supervised:
        model = SupervisedNeuralTopicModel(
            embedding_dim=args.embedding_dim,
            topic_dim=args.topic_dim,
            num_labels=num_labels,
            regression_mode=args.regression
        )
    else:
        model = NeuralTopicModel(
            embedding_dim=args.embedding_dim,
            topic_dim=args.topic_dim
        )

    # 训练和测试
    train_model(model, dataset, val_dataset=dataset_val, supervised=args.supervised, regression_mode=args.regression)
    # train_model(train_loader, model, device, args)

    # if args.supervised:
    #     test_model(test_loader, model, device)

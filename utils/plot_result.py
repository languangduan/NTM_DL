import re
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
def parse_logs(log_file_path):
    """
    从日志文件中提取所有训练和验证指标。

    Args:
        log_file_path (str): 日志文件路径。

    Returns:
        dict: 包含训练和验证指标的字典。
    """
    results = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "micro_f1": [],
        "macro_f1": [],
    }

    with open(log_file_path, 'r') as f:
        for line in f:
            # # 提取训练损失
            # train_loss_match = re.search(r"Train Epoch \d+/\d+, Loss: ([\d.]+)", line)
            # if train_loss_match:
            #     results["train_loss"].append(float(train_loss_match.group(1)))

            # 提取所有损失（训练和验证）
            loss_match = re.search(r"Loss: ([\d.]+)", line)
            if loss_match and "Pretrain Epoch" not in line:
                # 如果在 "Train Epoch" 或 "Validation Results" 后匹配到 Loss，分别判断
                if "Train" in line:
                    results["train_loss"].append(float(loss_match.group(1)))
                else:
                    results["val_loss"].append(float(loss_match.group(1)))

            # 提取验证准确率
            accuracy_match = re.search(r"Accuracy: ([\d.]+)", line)
            if accuracy_match:
                results["accuracy"].append(float(accuracy_match.group(1)))

            # 提取 Micro-F1
            micro_f1_match = re.search(r"Micro-F1: ([\d.]+)", line)
            if micro_f1_match:
                results["micro_f1"].append(float(micro_f1_match.group(1)))

            # 提取 Macro-F1
            macro_f1_match = re.search(r"Macro-F1: ([\d.]+)", line)
            if macro_f1_match:
                results["macro_f1"].append(float(macro_f1_match.group(1)))

    return results


def plot_metrics(results):
    """
    绘制训练和验证指标的变化趋势。

    Args:
        results (dict): 包含训练和验证指标的字典。
    """
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(12, 8))
    plt.suptitle('MovieReview Training and Validation Metrics Over Epochs', fontsize=25)  # 添加题头
    # 训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, results["train_loss"], label='Train Loss', marker='o')
    plt.plot(epochs, results["val_loss"], label='Validation Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=12)

    # 验证准确率
    plt.subplot(2, 2, 2)
    plt.plot(epochs, results["accuracy"], label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=12)

    # Micro-F1 分数
    plt.subplot(2, 2, 3)
    plt.plot(epochs, results["micro_f1"], label='Micro-F1', marker='o', color='green')
    plt.title('Micro-F1 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Micro-F1')
    plt.legend(fontsize=12)

    # Macro-F1 分数
    plt.subplot(2, 2, 4)
    plt.plot(epochs, results["macro_f1"], label='Macro-F1', marker='o', color='orange')
    plt.title('Macro-F1 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('movie.pdf')
    plt.show()


# 示例用法
log_file_path ="../20news.log"  # 替换为日志文件路径
results = parse_logs(log_file_path)
print(results)
plot_metrics(results)

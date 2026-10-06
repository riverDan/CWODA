import torch.nn.functional as F
import torch

# 对比学习损失函数：NT-Xent 损失
def contrastive_loss(features1, features2, temperature=0.5):
    """
    features1: a tensor of shape (N, d)
    features2: a tensor of shape (N, d)
    """
    # 首先将特征归一化
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    batch_size = features1.shape[0]

    # 拼接两个视图的特征得到 (2N, d)
    features = torch.cat([features1, features2], dim=0)  # shape (2N, d)

    # 计算相似度矩阵 (2N, 2N)
    similarity_matrix = torch.matmul(features, features.t())

    # 构造对比标签（正样本对索引，i与i+N为一组）
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(features.device)

    # 对每行除以温度项
    similarity_matrix = similarity_matrix / temperature

    # 屏蔽对角线位置的相似度
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
    similarity_matrix.masked_fill_(mask, -9e15)

    # 计算对比损失
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

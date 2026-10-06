import torch


# def compute_openmax(logits, alpha=10):
#     """
#     计算 OpenMax 的未知类别概率
#     :param logits: 原始输出 logits (batch_size, num_classes)
#     :param alpha: OpenMax 超参数，控制 Weibull 分布裁剪
#     :return: openmax_scores (batch_size, num_classes), unknown_scores (batch_size, 1)
#     """
#     batch_size, num_classes = logits.shape

#     # 软最大概率
#     softmax_probs = torch.nn.functional.softmax(logits, dim=1)

#     # 防止 alpha 超出类别数
#     alpha = min(alpha, num_classes)  # 确保 topk 不会超出类别数

#     # 计算 Weibull 调整因子
#     topk_values, _ = torch.topk(softmax_probs, k=alpha, dim=1)  # 取前 alpha 个最高的概率
#     weibull_factors = torch.exp(-topk_values)  # Weibull 退化因子

#     # OpenMax 调整
#     openmax_scores = softmax_probs.clone()  # 复制 softmax 结果
#     openmax_scores[:, :alpha] *= (1 - weibull_factors)  # 仅对 top-k 进行 Weibull 调整

#     # 计算未知类别的分数
#     unknown_scores = 1 - openmax_scores.sum(dim=1, keepdim=True)

#     return openmax_scores, unknown_scores

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import weibull_min


def compute_openmax(logits, means, weibull_params, alpha=10):
    """
    计算 OpenMax 的未知类别概率
    :param logits: 原始输出 logits (batch_size, num_classes)
    :param means: 训练集中每个类别的均值向量 (num_classes, feature_dim)
    :param weibull_params: Weibull 分布参数 (num_classes, 2), 其中包含 (lambda, kappa)
    :param alpha: OpenMax 超参数，控制 Weibull 分布裁剪
    :return: openmax_scores (batch_size, num_classes), unknown_scores (batch_size, 1)
    """
    batch_size, num_classes = logits.shape

    # 防止 alpha 超出类别数
    alpha = min(alpha, num_classes)

    # 计算 top-k 类别索引（按照 logits 降序）
    topk_values, topk_indices = torch.topk(logits, k=alpha, dim=1)

    # 初始化 ω（所有类别默认 1）
    omega = torch.ones_like(logits)

    selected_means = means[topk_indices]
    selected_params = weibull_params[topk_indices].clamp_min(1e-6)
    distances = torch.norm(logits[:, None, :] - selected_means, p=2, dim=2)
    lambdas = selected_params[:, :, 0]
    kappas = selected_params[:, :, 1]
    # Use the Weibull CDF so larger distances receive stronger unknown-class evidence.
    weibull_factor = 1.0 - torch.exp(-torch.pow(distances / lambdas, kappas))
    rank_weights = torch.arange(alpha, 0, -1, device=logits.device, dtype=logits.dtype) / float(alpha)
    omega_updates = 1 - rank_weights.view(1, -1) * weibull_factor
    omega.scatter_(1, topk_indices, omega_updates)

    # 修正 logits
    revised_logits = logits * omega

    # 计算未知类别得分 v̂_0(x)
    unknown_scores = torch.sum(logits * (1 - omega), dim=1, keepdim=True)

    # 计算 OpenMax 归一化概率
    joint_prob = torch.softmax(torch.cat([revised_logits, unknown_scores], dim=1), dim=1)
    openmax_scores = joint_prob[:, :num_classes]
    unknown_scores = joint_prob[:, num_classes:]

    return openmax_scores, unknown_scores


def compute_source_class_params(features, labels, num_classes, tail_size=20):
    """Calibrate logit-space distances using correctly classified SOURCE labels.

    Center j is explicitly class j. No target labels or arbitrary cluster IDs.
    """
    if features.ndim != 2 or features.shape[1] != num_classes:
        raise ValueError('Expected one logit coordinate per known source class')
    if not torch.isfinite(features).all():
        raise ValueError('Non-finite calibration logits')
    features = features.detach()
    labels = labels.to(features.device)
    correct = features.argmax(dim=1) == labels
    means, params = [], []
    for j in range(num_classes):
        samples = features[(labels == j) & correct]
        if len(samples) < 3:
            raise ValueError(f'Source calibration class {j}: only {len(samples)} correct examples')
        center = samples.mean(dim=0)
        distances = torch.linalg.vector_norm(samples - center, dim=1).cpu().numpy()
        tail = np.maximum(np.sort(distances)[-min(tail_size, len(distances)):], 1e-6)
        shape, _, scale = weibull_min.fit(tail, floc=0)
        if not np.isfinite([shape, scale]).all() or min(shape, scale) <= 0:
            raise ValueError(f'Invalid Weibull fit for source class {j}')
        means.append(center)
        params.append([scale, shape])
    return torch.stack(means), features.new_tensor(params)


def compute_means_and_weibull_params_clustered(features, num_classes, tail_size=20):
    """
    基于 K-Means 聚类计算类别均值 (means) 和 Weibull 参数 (weibull_params)
    :param features: 目标域样本特征 (num_samples, feature_dim)
    :param num_classes: 设定的伪类别数 (即 K-Means 的 K)
    :param tail_size: 用于拟合 Weibull 分布的尾部样本数
    :return: means (num_classes, feature_dim), weibull_params (num_classes, 2)
    """
    feature_dim = features.shape[1]

    # **关键修正点：确保 features 在 CPU 上**
    features_cpu = features.cpu().numpy()

    # 使用 K-Means 进行聚类
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(features_cpu)

    means = torch.zeros((num_classes, feature_dim)).to(features.device)  # 保持 device 一致
    weibull_params = torch.zeros((num_classes, 2)).to(features.device)  # (lambda, kappa)

    for j in range(num_classes):
        # 获取聚类 j 的样本
        cluster_mask = torch.tensor(pseudo_labels == j, dtype=torch.bool, device=features.device)
        cluster_samples = features[cluster_mask]
        if cluster_samples.shape[0] == 0:
            continue

        # 计算类别均值
        mean_j = torch.mean(cluster_samples, dim=0)
        means[j] = mean_j

        # 计算所有样本到均值的欧几里得距离
        distances = torch.norm(cluster_samples - mean_j, dim=1).cpu().numpy()

        # 选取 tail_size 个最大距离样本
        tail_distances = np.sort(distances)[-tail_size:]

        # 拟合 Weibull 分布，得到 λ (scale) 和 κ (shape)
        kappa, loc, lambda_ = weibull_min.fit(tail_distances, floc=0)
        weibull_params[j] = torch.tensor([lambda_, kappa], device=features.device)

    return means, weibull_params


import torch
from torch import nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, latent_dim)  # 均值头
        self.logvar_layer = nn.Linear(64, latent_dim)  # 方差头

        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # 采样潜在变量

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def train_vae(features, input_dim, latent_dim, num_epochs=50, learning_rate=1e-3):
    """
    训练 VAE 模型
    :param features: 目标域特征 (num_samples, feature_dim)
    :param input_dim: 输入特征维度
    :param latent_dim: 潜在空间维度
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :return: 训练好的 VAE 模型
    """
    vae = VariationalAutoEncoder(input_dim, latent_dim).to(features.device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        vae.train()
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(features)

        # 计算重构误差
        recon_loss = F.mse_loss(recon_x, features)

        # KL 散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return vae

def compute_vae_unknown_scores(features, vae, alpha=0.5):
    """
    使用 VAE 模型计算未知类别的概率分数
    :param features: 目标域特征 (num_samples, feature_dim)
    :param vae: 训练好的 VAE 模型
    :param alpha: 重构误差与潜在分布权重
    :return: 未知类别分数 (num_samples,)
    """
    vae.eval()
    recon_x, mu, logvar = vae(features)

    # 重构误差
    recon_error = torch.norm(features - recon_x, dim=1)

    # 潜在分布概率
    z = vae.reparameterize(mu, logvar)
    latent_prob = torch.exp(-0.5 * torch.sum(z ** 2, dim=1))

    # 综合分数
    unknown_scores = alpha * recon_error + (1 - alpha) * (1 - latent_prob)
    return unknown_scores


def compute_dear_scores(logits, target_vectors, num_classes):
    """
    计算 DEAR 算法的未知类别概率
    :param logits: 原始输出 logits (batch_size, num_classes)
    :param target_vectors: 目标域类别锚点向量 (num_classes, feature_dim)
    :return: known_probs (batch_size, num_classes), unknown_scores (batch_size, 1)
    """
    # 1. 计算狄利克雷证据参数
    alphas = torch.exp(logits) + 1

    # 2. 计算总不确定性
    S = torch.sum(alphas, dim=1, keepdim=True)  # 狄利克雷强度
    uncertainty = num_classes / S  # 总不确定性

    # 3. 计算已知类别概率
    known_probs = (alphas - 1) / S

    # 4. 计算未知类别概率（直接来自总不确定性）
    unknown_scores = uncertainty - torch.min(uncertainty)  # 数值稳定化
    unknown_scores = unknown_scores / torch.max(unknown_scores)  # 归一化到 [0,1]

    return known_probs, unknown_scores

def compute_class_anchors(features, num_classes, method='kmeans'):
    """
    计算类别锚点向量
    :param features: 目标域样本特征 (num_samples, feature_dim)
    :param num_classes: 类别数量（包括伪类别）
    :param method: 'kmeans' 或 'medoid'
    :return: 类别锚点向量 (num_classes, feature_dim)
    """
    features = features.detach().cpu().numpy()

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        kmeans.fit(features)
        anchors = torch.tensor(kmeans.cluster_centers_)

    elif method == 'medoid':
        from sklearn_extra.cluster import KMedoids
        kmedoids = KMedoids(n_clusters=num_classes, random_state=42)
        kmedoids.fit(features)
        anchors = torch.tensor(features[kmedoids.medoid_indices_])

    return anchors.to(features.device)


def compute_evidence(logits, use_softplus=False):
    """
    将 logits 转换为证据值，确保输出为非负数。
    参数：
    logits: 张量，形状为 (batch_size, num_classes)
    use_softplus: bool标记，True 时使用 softplus，否则使用 exp（默认 False）
    返回：
    evidence: 与 logits 形状一致，所有值均为非负
    """
    if use_softplus:
        evidence = F.softplus(logits)
    else:
        evidence = torch.exp(logits)
    return evidence

def compute_deep_evidential_probs(logits, use_softplus=False):
    """
    基于证据理论计算已知类别概率和未知类别（不确定性）概率。

    算法流程：
    1. 将 logits 转换为非负“证据”（通过 exp 或 softplus）
    2. 构造狄利克雷参数：α = evidence + 1
    3. 计算总证据 S = sum_j α_j
    4. 已知类别概率：p_known_j = (α_j - 1) / S
    5. 定义未知类别概率（不确定性指标）：u = num_classes / S
        当 S 小（证据不足）时，u 大，表示模型不确定性高。
    6. （可选）对 u 归一化至 [0,1] 范围。

    参数：
    logits: 张量，形状为 (batch_size, num_classes)
    use_softplus: 是否使用 softplus（默认 False）

    返回：
    known_probs: 形状 (batch_size, num_classes) 的已知类别预测概率
    unknown_scores: 形状 (batch_size, 1) 的未知类别概率（整体不确定性）
    """
    num_classes = logits.shape[1]
    # 1. 将 logits 转换为证据
    evidence = compute_evidence(logits, use_softplus)
    # 2. 构造狄利克雷参数: α = evidence + 1
    alphas = evidence + 1.0
    # 3. 总证据 S
    S = torch.sum(alphas, dim=1, keepdim=True)   # shape: (batch_size, 1)
    # 4. 已知类别概率
    known_probs = (alphas - 1.0) / S
    # 5. 不确定性/未知类别概率
    unknown_scores = num_classes / S
    # 可选：归一化 unknown_scores 到 [0,1]
    unknown_scores = unknown_scores / torch.max(unknown_scores)
    return known_probs, unknown_scores


# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# class GCNLayer(nn.Module):
#     """ 简单的 GCN 层 """
#     def __init__(self, in_features, out_features):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)

#     def forward(self, X, A):
#         """
#         X: 节点特征 (batch_size, feature_dim)
#         A: 归一化邻接矩阵 (batch_size, batch_size)
#         """
#         return F.relu(self.linear(torch.matmul(A, X)))

# def compute_adjacency_matrix(features, sigma=1.0, threshold=0.5):
#     """
#     计算邻接矩阵 (A) 采用高斯核函数或余弦相似度
#     :param features: (batch_size, feature_dim) 特征向量
#     :param sigma: 高斯核带宽
#     :param threshold: 阈值，去掉低相关的连接
#     :return: 归一化的邻接矩阵 (batch_size, batch_size)
#     """
#     device = features.device
#     batch_size = features.size(0)

#     # 计算余弦相似度
#     norm_features = F.normalize(features, p=2, dim=1)
#     adjacency_matrix = torch.mm(norm_features, norm_features.T).to(device)

#     # 应用阈值
#     adjacency_matrix = torch.where(adjacency_matrix > threshold, adjacency_matrix, torch.zeros_like(adjacency_matrix).to(device))

#     # 归一化 A (D^-1/2 A D^-1/2)
#     degree_matrix = torch.diag(adjacency_matrix.sum(dim=1)).to(device)
#     D_inv_sqrt = torch.inverse(torch.sqrt(degree_matrix + 1e-5)).to(device)
#     adjacency_matrix = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

#     return adjacency_matrix

# class OpenMaxGCN(nn.Module):
#     def __init__(self, feature_dim, num_classes, alpha=10):
#         super(OpenMaxGCN, self).__init__()
#         self.alpha = alpha
#         self.gcn_layer = GCNLayer(feature_dim, feature_dim)  # GCN
#         self.logits_proj = nn.Linear(num_classes, feature_dim)  # 变换 logits 形状

#     def forward(self, logits, features):
#         """
#         采用 GCN 增强 OpenMax 计算未知类别概率
#         :param logits: 原始输出 logits (batch_size, num_classes)
#         :param features: 目标域特征 (batch_size, feature_dim)
#         :return: openmax_scores (batch_size, num_classes), unknown_scores (batch_size, 1)
#         """
#         device = logits.device  # 确保 logits 和 features 在同一设备
#         batch_size, num_classes = logits.shape

#         # 构造邻接矩阵
#         adjacency_matrix = compute_adjacency_matrix(features).to(device)

#         # GCN 传播信息
#         enhanced_features = self.gcn_layer(features.to(device), adjacency_matrix)

#         # 修正 logits 形状
#         logits_transformed = self.logits_proj(logits)  # (batch_size, feature_dim)

#         # 计算增强后的 logits
#         enhanced_logits = logits + torch.mm(enhanced_features, logits_transformed.T).diag().unsqueeze(1)

#         # 计算 Weibull 调整因子
#         softmax_probs = F.softmax(enhanced_logits, dim=1)
#         topk_values, _ = torch.topk(softmax_probs, k=min(self.alpha, num_classes), dim=1)
#         weibull_factors = torch.exp(-topk_values)

#         # OpenMax 调整
#         openmax_scores = softmax_probs.clone()
#         openmax_scores[:, :self.alpha] *= (1 - weibull_factors)

#         # 计算未知类别概率
#         unknown_scores = 1 - openmax_scores.sum(dim=1, keepdim=True)

#         return openmax_scores, unknown_scores


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np
# # import time
# # import datetime
# # from torch_geometric.nn import GCNConv


# # class GCN(nn.Module):
# #     """Graph Convolutional Network for feature enhancement"""
# #     def __init__(self, in_dim, hidden_dim, out_dim):
# #         super(GCN, self).__init__()
# #         self.conv1 = GCNConv(in_dim, hidden_dim)
# #         self.conv2 = GCNConv(hidden_dim, out_dim)

# #     def forward(self, x, edge_index):
# #         x = self.conv1(x, edge_index)
# #         x = F.relu(x)
# #         x = self.conv2(x, edge_index)
# #         return x


# # def compute_graph_structure(features, k=5):
# #     """
# #     计算特征空间中的 k 近邻图
# #     :param features: shape (N, d)
# #     :param k: 近邻数量
# #     :return: edge_index (图的边)
# #     """
# #     from sklearn.neighbors import kneighbors_graph
# #     import torch_geometric.utils as pyg_utils

# #     adj_matrix = kneighbors_graph(features.cpu().numpy(), k, mode='connectivity', include_self=True)
# #     edge_index = torch.tensor(pyg_utils.from_scipy_sparse_matrix(adj_matrix)[0], dtype=torch.long)
# #     return edge_index


# # def compute_openmax_gcn(logits, features, edge_index, gcn_model):
# #     """
# #     计算 GCN 增强的 OpenMax 分数
# #     :param logits: (batch_size, num_classes)
# #     :param features: (batch_size, feature_dim)
# #     :param edge_index: (2, num_edges) 图结构
# #     :param gcn_model: 预训练的 GCN
# #     :return: openmax_scores, unknown_scores
# #     """
# #     enhanced_features = gcn_model(features, edge_index)
# #     logits += torch.mm(enhanced_features, nn.Parameter(torch.randn(features.shape[1], logits.shape[1], device=logits.device)))

# #     # 计算 OpenMax
# #     softmax_probs = F.softmax(logits, dim=1)
# #     alpha = min(10, logits.shape[1])
# #     topk_values, _ = torch.topk(softmax_probs, k=alpha, dim=1)
# #     weibull_factors = torch.exp(-topk_values)
# #     openmax_scores = softmax_probs.clone()
# #     openmax_scores[:, :alpha] *= (1 - weibull_factors)
# #     unknown_scores = 1 - openmax_scores.sum(dim=1, keepdim=True)

# #     return openmax_scores, unknown_scores



# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import numpy as np
# import time
# import datetime
# from torch.autograd import Variable
# from torch.nn import CrossEntropyLoss

# # In your `GCNLayer` or relevant layer:
# class GCNLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, X, A):
#         # Ensure A and X are on the same device
#         # device = X.device
#         # A = A.to(device)  # Move A to the same device as X

#         # Matrix multiplication on the same device
#         print(A.is_cuda,X.is_cuda)
#         return F.relu(self.linear(torch.matmul(A, X)))

# # Define the adjacency matrix computation
# def compute_adjacency_matrix(features, sigma=1.0, threshold=0.5):
#     """
#     Compute the adjacency matrix (A) using Gaussian kernel or cosine similarity.
#     :param features: (batch_size, feature_dim) Feature vectors
#     :param sigma: Bandwidth of the Gaussian kernel
#     :param threshold: Threshold to discard low-correlated connections
#     :return: Normalized adjacency matrix (batch_size, batch_size)
#     """
#     device = features.device
#     batch_size = features.size(0)

#     # Compute cosine similarity
#     norm_features = F.normalize(features, p=2, dim=1)
#     adjacency_matrix = torch.mm(norm_features, norm_features.T).to(device)

#     # Apply threshold
#     adjacency_matrix = torch.where(adjacency_matrix > threshold, adjacency_matrix, torch.zeros_like(adjacency_matrix).to(device))

#     # Normalize A (D^(-1/2) * A * D^(-1/2))
#     degree_matrix = torch.diag(adjacency_matrix.sum(dim=1)).to(device)
#     D_inv_sqrt = torch.inverse(torch.sqrt(degree_matrix + 1e-5)).to(device)
#     adjacency_matrix = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

#     return adjacency_matrix

# class OpenMaxGCN(nn.Module):
#     def __init__(self, feature_dim, num_classes, alpha=10):
#         super(OpenMaxGCN, self).__init__()
#         self.alpha = alpha
#         self.gcn_layer = GCNLayer(feature_dim, feature_dim)  # GCN
#         self.logits_proj = nn.Linear(num_classes, feature_dim)  # Transform logits shape

#     def forward(self, logits, features):
#         """
#         Enhance OpenMax calculation with GCN.
#         :param logits: Raw logits (batch_size, num_classes)
#         :param features: Target domain features (batch_size, feature_dim)
#         :return: openmax_scores (batch_size, num_classes), unknown_scores (batch_size, 1)
#         """
#         device = logits.device  # Ensure logits and features are on the same device
#         batch_size, num_classes = logits.shape

#         # Move features to the same device as logits
#         features = features.to(device)

#         # Compute adjacency matrix
#         adjacency_matrix = compute_adjacency_matrix(features).to(device)

#         # Propagate information using GCN
#         enhanced_features = self.gcn_layer(features, adjacency_matrix)

#         # Project logits to match feature dimension
#         logits_transformed = self.logits_proj(logits).to(device)  # Ensure logits are on the same device

#         # Compute enhanced logits
#         enhanced_logits = logits + torch.mm(enhanced_features, logits_transformed.T).diag().unsqueeze(1)

#         # Compute Weibull adjustment factors
#         softmax_probs = F.softmax(enhanced_logits, dim=1)
#         topk_values, _ = torch.topk(softmax_probs, k=min(self.alpha, num_classes), dim=1)
#         weibull_factors = torch.exp(-topk_values)

#         # OpenMax adjustment
#         openmax_scores = softmax_probs.clone()
#         openmax_scores[:, :self.alpha] *= (1 - weibull_factors)

#         # Compute unknown class probability
#         unknown_scores = 1 - openmax_scores.sum(dim=1, keepdim=True)

#         return openmax_scores, unknown_scores

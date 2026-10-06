import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import inspect
from sklearn.manifold import TSNE
import pandas as pd


_TSNE_ITER_PARAM = 'max_iter' if 'max_iter' in inspect.signature(TSNE.__init__).parameters else 'n_iter'


def build_tsne(max_iter, **kwargs):
    tsne_kwargs = dict(kwargs)
    tsne_kwargs[_TSNE_ITER_PARAM] = max_iter
    return TSNE(**tsne_kwargs)

def visualize_features(features, labels, feature_dir):
    """
    使用 t-SNE 可视化特征向量。
    features: ndarray, shape=(num_samples, num_features)
    labels: ndarray, shape=(num_samples,)
    feature_dir: 输出路径
    """
    # matplotlib.rcParams['font.family'] = 'Times New Roman'

    sns.set(style="whitegrid")

    # 自定义标签名
    label_names = ['UE1', 'UE2', 'UE3', 'UE4', 'UE5', 'UE6', 'UE7', 'unknown']

    # 将标签数字映射为字符串
    labels_named = [label_names[int(i)] for i in labels]

    # 使用 t-SNE 降维
    tsne = build_tsne(n_components=2, perplexity=20, max_iter=300)
    features_tsne = tsne.fit_transform(features)

    # 创建 dataframe 方便 seaborn 处理
    df = pd.DataFrame({
        "x": features_tsne[:, 0],
        "y": features_tsne[:, 1],
        "label": labels_named
    })

    unique_labels = sorted(set(labels_named), key=lambda x: label_names.index(x))
    palette = sns.color_palette("tab10", len(unique_labels))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=dict(zip(unique_labels, palette)),
        s=20,
        edgecolor='none',
        alpha=0.8
    )

    # plt.xlabel("t-SNE dimension 1", fontsize=12, fontname='Times New Roman')
    # plt.ylabel("t-SNE dimension 2", fontsize=12, fontname='Times New Roman')
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(loc='upper right', frameon=True, prop={'size': 10})
    plt.tight_layout()
    plt.savefig(osp.join(feature_dir + '.pdf'), dpi=300, format="pdf")
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from sklearn.manifold import TSNE

def visualize_features_two_sets(features_source, labels_source, features_target, labels_target, feature_dir):
    """
    t-SNE 可视化 Source + Target 特征
    灰色 - Source
    蓝色 - Target Known (labels_target != 7)
    红色 - Target Unknown (labels_target == 7)

    features_source: ndarray, (num_source, feat_dim)
    labels_source: ndarray, (num_source,)
    features_target: ndarray, (num_target, feat_dim)
    labels_target: ndarray, (num_target,)
    feature_dir: 保存路径（不带后缀）
    """

    # 合并数据
    features = np.vstack([features_source, features_target])

    # 构建三类可视化标签
    combined_labels = np.zeros(len(features), dtype=int)  # 默认 0=source
    # Target 范围
    target_idx_start = len(features_source)
    target_labels = labels_target

    # target known → 1, target unknown → 2
    target_known_mask = target_labels != 7
    combined_labels[target_idx_start:][target_known_mask] = 1  # 蓝色
    combined_labels[target_idx_start:][~target_known_mask] = 2  # 红色

    # t-SNE
    tsne = build_tsne(n_components=2, perplexity=20, max_iter=500, init="pca", random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 配色 & 名称
    colors = {
        0: (0.0, 0.0, 1.0),  # 蓝色
        1: (1.0, 0.0, 0.0),  # 红色
        2: (0.6, 0.6, 0.6)   # 灰色
    }
    labels_name = {
        0: 'source',
        1: 'target known',
        2: 'target unknown'
    }

    # 绘制
    plt.figure(figsize=(6, 5))
    for label_value in np.unique(combined_labels):
        idx = combined_labels == label_value
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1],
                    s=8, c=[colors[label_value]], label=labels_name[label_value], alpha=0.8, linewidths=0)

    # 论文风格
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()

    # 保存
    plt.savefig(osp.join(feature_dir + '.pdf'), dpi=300, format="pdf")
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os.path as osp
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

def lighten_color(color, amount=0.5):
    """
    让颜色变浅，amount=0~1，越大越接近白色
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = mcolors.to_rgb(mcolors.CSS4_COLORS[color])
    return (1 - amount) + amount * np.array(c)

def visualize_features_multi_class_light_dark(features_source, labels_source,
                                               features_target, labels_target,
                                               feature_dir, unknown_class_id=7):
    """
    t-SNE 可视化：
        - 同一类别颜色一致
        - Source 浅色
        - Target 深色
        - Target Unknown 黑色
    """
    num_classes = len(np.unique(labels_source))

    # 合并特征用于 t-SNE
    features = np.vstack([features_source, features_target])
    tsne = build_tsne(n_components=2, perplexity=20, max_iter=1000, init="pca", random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 分离 Source / Target 坐标
    source_tsne = features_tsne[:len(features_source)]
    target_tsne = features_tsne[len(features_source):]

    # 定义颜色（深色用于 Target，浅色用于 Source）
    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i % 10) for i in range(num_classes)]
    source_colors = [lighten_color(c, 0.5) for c in base_colors]  # 浅色版
    target_colors = base_colors  # 深色版
    unknown_color = (0, 0, 0)  # 黑色

    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Nimbus Roman'

    # 画已知类别
    for cls in range(num_classes):
        idx_s = (labels_source == cls)
        plt.scatter(source_tsne[idx_s, 0], source_tsne[idx_s, 1],
                    s=8, c=[source_colors[cls]], label=f"S-{cls}",
                    alpha=0.8, linewidths=0)

        idx_t = (labels_target == cls)
        plt.scatter(target_tsne[idx_t, 0], target_tsne[idx_t, 1],
                    s=8, c=[target_colors[cls]], label=f"T-{cls}",
                    alpha=0.8, linewidths=0)

    # 画 Target Unknown
    idx_tu = (labels_target == unknown_class_id)
    if np.sum(idx_tu) > 0:
        plt.scatter(target_tsne[idx_tu, 0], target_tsne[idx_tu, 1],
                    s=8, c=[unknown_color], label="T-Unknown",
                    alpha=0.8, linewidths=0)


    # # 创建 handles 列表（每一列放 T-i 和 S-i，最后一列是 T-未知）
    # handles = []
    # labels = []
    # for i in range(num_classes):
    #     handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=target_colors[i], markersize=8))
    #     labels.append(f"T-UE{i+1}")
    #     handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=source_colors[i], markersize=8))
    #     labels.append(f"S-UE{i+1}")

    # # 添加 Unknown 类
    # handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=unknown_color, markersize=8))
    # labels.append("T-Unknown")

    # fig, ax = plt.subplots(figsize=(8, 1.0))
    # ax.axis('off')  # 隐藏坐标轴
    # # 绘制图例
    # legend = ax.legend(handles, labels,
    #                 loc='center',
    #                 frameon=True,
    #                 ncol=num_classes+1,        # 列数 = 类别数 + 1 (未知)
    #                 fontsize=8,
    #                 columnspacing=1.0,
    #                 handletextpad=0.4)
    # plt.savefig(osp.join('legend.pdf'), dpi=300, format="pdf")
    # plt.close()

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # 保存 PDF
    plt.savefig(osp.join(feature_dir + '.pdf'), dpi=300, format="pdf")
    plt.close()



# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os.path as osp

# def visualize_features(features, labels, feature_dir):

#     """
#     使用 t-SNE 可视化特征向量。

#     参数:
#     features -- 特征向量，形状为 (num_samples, num_features)
#     labels -- 类别标签，形状为 (num_samples,)
#     epoch -- 当前训练轮次，用于标题显示
#     """
#     sns.set(style="whitegrid")

#     # 确保 features 和 labels 的形状匹配
#     # assert len(features) == len(labels), "Features and labels must have the same length."
#     # assert features.ndim == 2, "Features should be a 2D array."

#     # 使用 t-SNE 进行降维
#     # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     tsne = TSNE(n_components=2, perplexity=20, n_iter=300)
#     features_tsne = tsne.fit_transform(features)

#     # 可视化
#     plt.figure(figsize=(10, 8))
#     num_classes = len(set(labels))
#     palette = sns.color_palette("hls", num_classes)
#     sns.scatterplot(
#         x=features_tsne[:,0],
#         y=features_tsne[:,1],
#         hue=labels,
#         palette=palette,
#         s=15,
#         legend='full',
#         edgecolor='none'
#         )
#     # plt.title(f"t-SNE Visualization (Epoch {epoch})")
#     # plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.legend(title="Class", loc='upper right', frameon=True)
#     plt.tight_layout()
#     plt.show()

#     save_name = osp.join(feature_dir + '.svg')
#     plt.savefig(save_name, dpi=300,format="svg")
#     plt.close()

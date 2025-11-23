from options import args_parser
from Non_IID_bench import partition_data_non_iid
from cifar_load import get_dataloader

import torch
import torch.nn as nn
import torchvision.models as models

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import os

# 1. 提取特征，支持 GPU 加速
def extract_features(dataloader, model, device='cpu'):
    features = []
    labels = []

    model.to(device)  # 将模型移动到 GPU 或 CPU

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁止梯度计算
        for _, images, label_batch in dataloader:
            images = images.to(device)  # 将图像数据移动到 GPU 或 CPU
            # 提取特征 h (base-encoder 输出)
            h, _, _ = model(images)  # 只需要提取 h 部分作为特征
            h = h.cpu()  # 将特征移回 CPU 以进行后续处理

            features.append(h)
            labels.append(label_batch.cpu())  # 将标签移回 CPU

    # 将列表转换为张量
    features = torch.cat(features)  # 将列表中的张量拼接成一个大张量
    labels = torch.cat(labels)

    return features, labels


def visualize_and_save_features(features, labels=None, method='pca', client_idx=None, save_dir="visualizations"):
    # 针对不同的可视化方法创建子文件夹
    method_dir = os.path.join(save_dir, method)
    os.makedirs(method_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    
    # 选择降维方法
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)  # 针对 CIFAR-10 适当调整参数
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, random_state=42)

    # 将 PyTorch Tensor 转换为 Numpy 数组以用于降维算法
    reduced_features = reducer.fit_transform(features.numpy())

    # 可视化特征分布
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=20)
        plt.colorbar(scatter)
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=20)
    
    plt.title(f"Feature Distribution - Client {client_idx} ({method.upper()})")
    plt.grid(False)

    # 将结果保存到方法对应的文件夹中
    save_path = os.path.join(method_dir, f"client_{client_idx}_features_{method}.png")
    plt.savefig(save_path, bbox_inches='tight')  # 保存为PNG文件
    plt.close()  # 关闭当前图，防止显示重叠
    print(f"Client {client_idx} feature distribution saved to {save_path}")



# 主代码部分
args = args_parser()

X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data_non_iid(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

from networks.models import ModelFedCon
net_glob = ModelFedCon(args.model, args.out_dim, n_classes=10)
checkpoint = torch.load('warmup/SVHN.pth')
net_glob.load_state_dict(checkpoint['state_dict'])

# 3. 设置 GPU/CPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for client_idx in range(10):
    noise_level = args.noise
    if client_idx == args.n_parties - 1:
        noise_level = 0
    noise_level = args.noise / (args.n_parties - 1) * client_idx
    train_dl_local, _ = get_dataloader(args, X_train[net_dataidx_map[client_idx]], y_train[net_dataidx_map[client_idx]], 
                                       args.dataset, args.datadir, args.batch_size, is_labeled=True,
                                       data_idxs=net_dataidx_map[client_idx], pre_sz=args.pre_sz, input_sz=args.input_sz, 
                                       noise_level=noise_level)
    
    # 提取特征并放在 GPU 上处理
    client_features, client_labels = extract_features(train_dl_local, net_glob, device=device)
    
    # 选择方法 (tsne, pca, umap) 并可视化
    visualize_and_save_features(client_features, client_labels, method='umap', client_idx=client_idx)


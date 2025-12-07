from math import sqrt
import pandas as pd
from datasets import CIFAR10_truncated, SVHN_truncated, CIFAR100_truncated
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import logging
import random
from dataloaders import dataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(
        datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    SVHN_train_ds = SVHN_truncated(
        datadir, split='train', download=True, transform=transform)
    SVHN_test_ds = SVHN_truncated(
        datadir, split='test', download=True, transform=transform)

    X_train, y_train = SVHN_train_ds.data, SVHN_train_ds.target
    X_test, y_test = SVHN_test_ds.data, SVHN_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_skin_data(datadir, train_idxs, test_idxs):  # idxs相对所有data
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    all_data_path = 'med_classify_dataset/HAM10000_metadata'
    all_data_df = pd.read_csv(all_data_path)
    all_data_df = pd.concat(
        [all_data_df['image_id'], all_data_df['dx']], axis=1)

    X_train, y_train, X_test, y_test = [], [], [], []
    train_df = all_data_df.iloc[train_idxs]
    test_df = all_data_df.iloc[test_idxs]

    train_names = all_data_df.iloc[train_idxs]['image_id'].values.astype(
        str).tolist()
    train_lab = all_data_df.iloc[train_idxs]['dx'].values.astype(str)
    test_names = all_data_df.iloc[test_idxs]['image_id'].values.astype(
        str).tolist()
    test_lab = all_data_df.iloc[test_idxs]['dx'].values.astype(str)

    for idx in range(len(train_idxs)):
        X_train.append(datadir + train_names[idx] + '.jpg')
        y_train.append(CLASS_NAMES.index(train_lab[idx]))

    for idx in range(len(test_idxs)):
        X_test.append(datadir + test_names[idx] + '.jpg')
        y_test.append(CLASS_NAMES.index(test_lab[idx]))
    return X_train, y_train, X_test, y_test


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, labeled_num, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    state = np.random.get_state()
    np.random.shuffle(X_train)
    # print(a)
    # result:[6 4 5 3 7 2 0 1 8 9]
    np.random.set_state(state)
    np.random.shuffle(y_train)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        # min_require_size = 100
        sup_size = int(len(y_train) / 10)
        N = y_train.shape[0] - sup_size
        net_dataidx_map = {}
        for sup_i in range(labeled_num):
            net_dataidx_map[sup_i] = [i for i in range(
                sup_i * sup_size, (sup_i + 1) * sup_size)]

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties - labeled_num)]
            for k in range(K):
                idx_k = np.where(
                    y_train[int(labeled_num * len(y_train) / 10):] == k)[0] + sup_size
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(
                    [p * (len(idx_j) < N / (n_parties - labeled_num)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties - labeled_num):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j + labeled_num] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(
        y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def partition_data_allnoniid(dataset, datadir, train_idxs=None, test_idxs=None, partition="noniid", n_parties=10,
                             beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'SVHN':
        X_train, y_train, X_test, y_test = load_SVHN_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'skin':
        X_train, y_train, X_test, y_test = load_skin_data(
            datadir, train_idxs, test_idxs)

    if dataset != 'skin':
        n_train = y_train.shape[0]
        if partition == "homo" or partition == "iid":
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, n_parties)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

        elif partition == "noniid-labeldir" or partition == "noniid":
            min_size = 0
            min_require_size = 10
            K = 10

            N = y_train.shape[0]
            net_dataidx_map = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(n_parties)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(beta, n_parties))
                    proportions = np.array(
                        [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) *
                                   len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j,
                                 idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_parties):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            traindata_cls_counts = record_net_data_stats(
                y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
    else:
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


class ClientBasedNoiseGenerator:
    """基于客户端ID的异构噪声生成器 - 与 FedPAC 保持一致"""

    def __init__(self, num_clients, base_noise_level=0.05, max_noise_level=0.20, deterministic_pattern=True):
        """
        初始化噪声生成器

        Args:
            num_clients: 联邦学习中客户端总数
            base_noise_level: 基础噪声水平(最小噪声)
            max_noise_level: 最大噪声水平
            deterministic_pattern: 是否为每个客户端分配确定的噪声模式
        """
        self.num_clients = num_clients
        self.base_noise_level = base_noise_level
        self.max_noise_level = max_noise_level
        self.deterministic_pattern = deterministic_pattern

        # 存储手动设置的客户端噪声级别
        self.custom_noise_levels = {
            0: 0.02,  # 客户端0 - 低噪声
            1: 0.08,  # 客户端1 - 中等
            # 2: 0.18   # 客户端2 - 高噪声
        }

        # 验证自定义噪声级别的客户端ID
        for client_id in self.custom_noise_levels.keys():
            if client_id < 0 or client_id >= num_clients:
                logger.warning(
                    f"Invalid client_id {client_id} in custom_noise_levels. "
                    f"Valid range: [0, {num_clients-1}]. Ignoring this entry."
                )

        # 为每个客户端预先确定噪声类型偏好，使其在训练过程中保持一致
        if deterministic_pattern:
            self.client_noise_patterns = self._initialize_client_patterns()

    def _initialize_client_patterns(self):
        """为每个客户端初始化固定的噪声特性"""
        patterns = {}
        for client_id in range(self.num_clients):
            # 确定该客户端的主要噪声类型
            r = random.random()
            if r < 0.25:
                primary_noise = "gaussian"
            elif r < 0.5:
                primary_noise = "spatial"
            elif r < 0.75:
                primary_noise = "saltpepper"
            else:
                primary_noise = "localized"

            # 确定该客户端的噪声强度
            if client_id in self.custom_noise_levels:
                # 如果有自定义噪声级别,直接跳过noise_factor计算
                # 因为 _get_noise_level() 会直接返回 custom_noise_levels 中的值
                noise_factor = None  # 标记为自定义,不使用插值计算
            else:
                # 客户端ID越大,噪声水平越高,但加入一些随机性避免完全线性关系
                base_factor = client_id / \
                    (self.num_clients - 1) if self.num_clients > 1 else 0.5
                random_factor = random.uniform(-0.1, 0.1)  # 添加±10%的随机波动
                noise_factor = min(max(base_factor + random_factor, 0.0), 1.0)

            patterns[client_id] = {
                "primary_noise": primary_noise,
                "noise_factor": noise_factor,
                # 为不同噪声类型预设额外参数
                "region_size": random.uniform(0.2, 0.5) if primary_noise == "localized" else None,
                "spatial_variation": random.uniform(0.02, 0.08) if primary_noise == "spatial" else None
            }
        return patterns

    def _get_noise_level(self, client_id):
        """根据客户端ID获取噪声等级"""
        # 优先使用自定义设置的噪声级别
        if client_id in self.custom_noise_levels:
            return self.custom_noise_levels[client_id]

        if self.deterministic_pattern:
            noise_factor = self.client_noise_patterns[client_id]["noise_factor"]
        else:
            # 按照客户端ID线性分配噪声强度
            noise_factor = client_id / \
                (self.num_clients - 1) if self.num_clients > 1 else 0.5

        # 在基础噪声和最大噪声之间插值
        return self.base_noise_level + (self.max_noise_level - self.base_noise_level) * noise_factor

    def add_gaussian_noise(self, x, client_id, mean=0.0, noise_level=None):
        """添加高斯噪声"""
        if noise_level is None:
            noise_level = self._get_noise_level(client_id)

        noise = torch.randn_like(x) * noise_level + mean
        return x + noise

    def add_salt_pepper_noise(self, x, client_id, noise_level=None):
        """添加椒盐噪声"""
        if noise_level is None:
            noise_level = self._get_noise_level(client_id) * 1.5  # 椒盐噪声稍强

        x_noisy = x.clone()
        # 盐噪声 (白点)
        salt_mask = torch.rand_like(x) < (noise_level / 2)
        x_noisy[salt_mask] = 1.0
        # 椒噪声 (黑点)
        pepper_mask = torch.rand_like(x) < (noise_level / 2)
        x_noisy[pepper_mask] = 0.0

        return x_noisy

    def add_localized_noise(self, x, client_id, noise_level=None, region_size=None):
        """添加局部区域噪声"""
        if noise_level is None:
            noise_level = self._get_noise_level(client_id) * 2.0  # 局部噪声更强

        if region_size is None and self.deterministic_pattern:
            region_size = self.client_noise_patterns[client_id].get(
                "region_size", 0.3)
        elif region_size is None:
            region_size = 0.2 + 0.3 * (client_id / (self.num_clients - 1))

        c, h, w = x.shape

        # 随机选择噪声区域的位置和大小
        region_h = int(h * region_size)
        region_w = int(w * region_size)

        top = random.randint(0, h - region_h) if h > region_h else 0
        left = random.randint(0, w - region_w) if w > region_w else 0

        # 创建噪声掩码
        mask = torch.zeros_like(x)
        mask[:, top:top+region_h, left:left+region_w] = 1.0

        # 生成并应用噪声
        noise = torch.randn_like(x) * noise_level * mask
        return x + noise

    def add_spatial_varying_noise(self, x, client_id, base_level=None, variation=None):
        """添加空间变化的噪声"""
        if base_level is None:
            base_level = self._get_noise_level(client_id)

        if variation is None and self.deterministic_pattern:
            variation = self.client_noise_patterns[client_id].get(
                "spatial_variation", base_level/2)
        elif variation is None:
            variation = base_level / 2

        # 创建空间变化的噪声掩码
        c, h, w = x.shape
        # 生成低频率的随机场，表示噪声强度的空间变化
        noise_map_size = max(h//8, 1), max(w//8, 1)
        noise_map = torch.rand(c, *noise_map_size)
        # 上采样到原始分辨率
        if noise_map_size[0] < h or noise_map_size[1] < w:
            noise_map = F.interpolate(noise_map.unsqueeze(
                0), size=(h, w), mode='bilinear')[0]

        # 基于噪声图生成每个像素的噪声水平
        pixel_noise_level = base_level + variation * noise_map

        # 为每个像素添加不同强度的噪声
        noise = torch.randn_like(x) * pixel_noise_level
        return x + noise

    def add_heterogeneous_noise(self, x, client_id):
        """根据客户端ID决定使用哪种噪声类型"""
        if self.deterministic_pattern:
            # 使用预设的噪声模式
            pattern = self.client_noise_patterns[client_id]
            primary_noise = pattern["primary_noise"]

            # 有10%的概率使用随机噪声类型，而不是主要类型
            if random.random() < 0.1:
                choices = ["gaussian", "spatial", "saltpepper", "localized"]
                choices.remove(primary_noise)
                primary_noise = random.choice(choices)

            if primary_noise == "gaussian":
                return self.add_gaussian_noise(x, client_id)
            elif primary_noise == "spatial":
                return self.add_spatial_varying_noise(x, client_id)
            elif primary_noise == "saltpepper":
                return self.add_salt_pepper_noise(x, client_id)
            else:  # localized
                return self.add_localized_noise(x, client_id)
        else:
            # 动态决定噪声类型，基于客户端ID
            client_ratio = client_id / \
                (self.num_clients - 1) if self.num_clients > 1 else 0.5
            gaussian_ratio = 0.7 - 0.4 * client_ratio
            sp_ratio = 0.1 + 0.3 * client_ratio
            local_ratio = 0.1 + 0.1 * client_ratio

            r = random.random()
            if r < gaussian_ratio:
                return self.add_gaussian_noise(x, client_id)
            elif r < gaussian_ratio + sp_ratio:
                return self.add_salt_pepper_noise(x, client_id)
            elif r < gaussian_ratio + sp_ratio + local_ratio:
                return self.add_localized_noise(x, client_id)
            else:
                return self.add_spatial_varying_noise(x, client_id)


# 创建一个全局的噪声生成器实例，与 FedPAC 保持一致
noise_generator = ClientBasedNoiseGenerator(
    num_clients=10, base_noise_level=0.05, max_noise_level=2.0, deterministic_pattern=False)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., client_id=0):
        self.std = std
        self.mean = mean
        self.client_id = client_id

    def __call__(self, tensor):
        return noise_generator.add_gaussian_noise(tensor, self.client_id)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, client_id={2})'.format(self.mean, self.std, self.client_id)


def get_dataloader(args, data_np, label_np, dataset_type, datadir, train_bs, is_labeled=None, data_idxs=None,
                   is_testing=False, pre_sz=40, input_sz=32, noise_level=0, client_id=0):
    if dataset_type == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                         std=[0.19803012, 0.20101562, 0.19703614])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'cifar10':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                         std=[0.24703233, 0.24348505, 0.26158768])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'skin':
        normalize = transforms.Normalize(mean=[0.7630332, 0.5456457, 0.57004654],
                                         std=[0.14092809, 0.15261231, 0.16997086])
    elif dataset_type == 'generated':
        # ds_gene = dataset.Generated(datadir, data_idxs, train=True)
        # dl_gene = data.DataLoader(dataset=ds_gene, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8)
        pass
    if not is_testing:
        if is_labeled:
            trans = transforms.Compose(
                [transforms.RandomCrop(size=(input_sz, input_sz)),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ToTensor(),
                 AddGaussianNoise(0., noise_level, client_id),
                 normalize
                 ])
            ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz, lab_trans=trans,
                                         is_labeled=True, is_testing=False)
        else:
            # 与 FedPAC 保持一致：使用弱增强 + 强增强
            weak_trans = transforms.Compose([
                transforms.RandomCrop(size=(input_sz, input_sz)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, client_id),
                normalize
            ])
            # 强增强：与 FedPAC 保持一致
            strong_trans = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=(input_sz, input_sz)),  # 更大范围的裁剪
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    # 随机调整亮度、对比度、饱和度和色调
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,
                ),  # 以较高概率应用颜色抖动
                transforms.RandomGrayscale(p=0.2),  # 20% 概率将图像转为灰度
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, client_id),
                normalize
            ])

            ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz,
                                         un_trans_wk=dataset.TransformTwice(
                                             weak_trans, strong_trans),
                                         data_idxs=data_idxs,
                                         is_labeled=False,
                                         is_testing=False)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs,
                             drop_last=False, shuffle=True, num_workers=8)
    else:
        if dataset_type == 'generated':
            pass
        else:
            ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, input_sz, input_sz, lab_trans=transforms.Compose([
                # K.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ]), is_labeled=True, is_testing=True)
            dl = data.DataLoader(
                dataset=ds, batch_size=train_bs, drop_last=False, shuffle=False, num_workers=8)
    return dl, ds

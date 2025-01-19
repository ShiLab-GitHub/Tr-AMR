from PIL import Image
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

# resnet结构2018数据集
class MyDataset(Dataset):
    def __init__(self, images_path, mode):
        self.images_path = images_path
        self.mode = mode
        X_train, X_test, Y_train, Y_test = self.read_data()
        if mode == "train":
            self.data = X_train
            self.label = Y_train
            self.snr = X_train
        if mode == "test":
            self.data = X_test
            self.label = Y_test
            self.snr = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # print(self.label[item], self.label[item].shape)
        data = self.data[item]  # (1024,2)
        # label = self.to_onehot(self.label[item][0])  # 24
        label = self.label[item]
        snr = self.snr[item]
        return data, label

    # def to_onehot(yy):
    #     mod = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
    #            '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK',
    #            'OQPSK']
    #     for index, nums in enumerate(mod):
    #         print(index, nums)
    #         if nums == yy:
    #             # yy1 = torch.nn.functional.one_hot(index)
    #             yy1 = index
    #             break
    #     return yy1

    def read_data(self):
        for i in range(0, 24):  # 24个数据集文件
            ########打开文件#######
            filename = './data/RML2018_modu_10_26/part' + str(i) + '.h5'
            print(filename)
            f = h5py.File(filename, 'r')
            ########读取数据#######
            X_data = f['X'][:]
            Y_data = f['Y'][:]
            Z_data = f['Z'][:]
            f.close()
            #########分割训练集和测试集#########
            # 每读取到一个数据文件就直接分割为训练集和测试集，防止爆内存
            n_examples = X_data.shape[0]
            n_train = int(n_examples * 0.7)  # 70%训练样本
            train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)  # 随机选取训练样本下标
            test_idx = list(set(range(0, n_examples)) - set(train_idx))  # 测试样本下标
            if i == 0:
                X_train = X_data[train_idx]
                Y_train = Y_data[train_idx]
                Z_train = Z_data[train_idx]
                X_test = X_data[test_idx]
                Y_test = Y_data[test_idx]
                Z_test = Z_data[test_idx]
            else:
                X_train = np.vstack((X_train, X_data[train_idx]))
                Y_train = np.vstack((Y_train, Y_data[train_idx]))
                Z_train = np.vstack((Z_train, Z_data[train_idx]))
                X_test = np.vstack((X_test, X_data[test_idx]))
                Y_test = np.vstack((Y_test, Y_data[test_idx]))
                Z_test = np.vstack((Z_test, Z_data[test_idx]))
        print('训练集X维度：', X_train.shape)
        print('训练集Y维度：', Y_train.shape)
        print('训练集Z维度：', Z_train.shape)
        print('测试集X维度：', X_test.shape)
        print('测试集Y维度：', Y_test.shape)
        print('测试集Z维度：', Z_test.shape)
        # h5file = h5py.File(self.images_path, 'r+')
        # X = np.array(h5file['X'][:])
        # Y = np.array(h5file['Y'][:])
        # np.random.seed(2018)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
        # n_examples = X.shape[0]
        # n_train = n_examples * 0.7
        # train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        # test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
        # X_train = X[train_idx]
        # X_test = X[test_idx]
        # Y_train = Y[train_idx]
        # Y_test = Y[test_idx]
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        images, labels = tuple(zip(*batch))
        images = torch.tensor(np.array(images)).unsqueeze(1)  # [8,1,1024,2]
        labels = torch.tensor(np.array(labels))  # [8,24]
        # images = images.permute(0,3,1,2)  # 128, 2, 1, 1024
        # max_tensor, _ = torch.max(images, 1, keepdim=True)  # [8,2]
        # images = images/max_tensor  # 做一下归一化

        return images, labels

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

# 修改的vit结构2018数据集训练
class MyGLUDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = self.read_data()
        if mode == "train":
            self.data = X_train
            self.label = Y_train
            self.snr = X_train
        if mode == "test":
            self.data = X_test
            self.label = Y_test
            self.snr = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # print(self.label[item], self.label[item].shape)
        data = self.data[item]  # (1024,2)
        # label = self.to_onehot(self.label[item][0])  # 24
        label = self.label[item]
        # snr = self.snr[item]
        return data, label

    def read_data(self):
        for i in range(0, 24):  # 24个数据集文件
            ########打开文件#######
            filename = './data/RML2018_modu_10_26/part' + str(i) + '.h5'
            print(filename)
            f = h5py.File(filename, 'r')
            ########读取数据#######
            X_data = f['X'][:]
            Y_data = f['Y'][:]
            Z_data = f['Z'][:]
            f.close()
            #########分割训练集和测试集#########
            # 每读取到一个数据文件就直接分割为训练集和测试集，防止爆内存
            n_examples = X_data.shape[0]
            n_train = int(n_examples * 0.7)  # 70%训练样本
            train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)  # 随机选取训练样本下标
            test_idx = list(set(range(0, n_examples)) - set(train_idx))  # 测试样本下标
            if i == 0:
                X_train = X_data[train_idx]
                Y_train = Y_data[train_idx]
                Z_train = Z_data[train_idx]
                X_test = X_data[test_idx]
                Y_test = Y_data[test_idx]
                Z_test = Z_data[test_idx]
            else:
                X_train = np.vstack((X_train, X_data[train_idx]))
                Y_train = np.vstack((Y_train, Y_data[train_idx]))
                Z_train = np.vstack((Z_train, Z_data[train_idx]))
                X_test = np.vstack((X_test, X_data[test_idx]))
                Y_test = np.vstack((Y_test, Y_data[test_idx]))
                Z_test = np.vstack((Z_test, Z_data[test_idx]))
        print('训练集X维度：', X_train.shape)
        print('训练集Y维度：', Y_train.shape)
        print('训练集Z维度：', Z_train.shape)
        print('测试集X维度：', X_test.shape)
        print('测试集Y维度：', Y_test.shape)
        print('测试集Z维度：', Z_test.shape)

        return X_train, X_test, Y_train, Y_test, Z_train, Z_test

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        images, labels = tuple(zip(*batch))
        images = torch.tensor(np.array(images)).unsqueeze(1)  # [8,1,1024,2]
        images = torch.tensor(np.array(images)) # [8,1,1024,2]
        labels = torch.tensor(np.array(labels))  # [8,24]
        # print("确认图像和标签的shape", images.shape, labels.shape)
        images = images.permute(0, 1, 3, 2)  # 8,1,2,1024
        # max_tensor, _ = torch.max(images, 1, keepdim=True)  # [8,2]
        # images = images/max_tensor  # 做一下归一化

        return images, labels

# 修改的vit结构2018数据集测试跑图
class MyGLUDatasetTest(Dataset):
    def __init__(self,  snr):
        self.snr = snr
        X_test, Y_test = self.read_data(self.snr)

        self.data = X_test
        self.label = Y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # print(self.label[item], self.label[item].shape)
        data = self.data[item]  # (1024,2)
        # label = self.to_onehot(self.label[item][0])  # 24
        label = self.label[item]
        return data, label

    def read_data(self, snr):
        # for i in range(0, 24):  # 24个数据集文件
        ########打开文件#######
        filename = './data/RML2018_snr/part' + str((snr+20)//2) + '.h5'  # 0到26，26个snr
        print(filename)
        f = h5py.File(filename, 'r')
        ########读取数据#######
        X_data = f['X'][:]
        Y_data = f['Y'][:]
        f.close()
        #########分割训练集和测试集#########
        # 每读取到一个数据文件就直接分割为训练集和测试集，防止爆内存
        n_examples = X_data.shape[0]
        n_train = int(n_examples * 0.7)  # 70%训练样本
        train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)  # 随机选取训练样本下标
        test_idx = list(set(range(0, n_examples)) - set(train_idx))  # 测试样本下标

        X_test = X_data[test_idx]
        Y_test = Y_data[test_idx]

        print('测试集X维度：', X_test.shape)
        print('测试集Y维度：', Y_test.shape)

        return X_test, Y_test

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        images, labels = tuple(zip(*batch))
        images = torch.tensor(np.array(images)).unsqueeze(1)  # [8,1,1024,2]
        images = torch.tensor(np.array(images)) # [8,1,1024,2]
        labels = torch.tensor(np.array(labels))  # [8,24]
        # print("确认图像和标签的shape", images.shape, labels.shape)
        images = images.permute(0,1,3,2)  # 8,1,2,1024
        # max_tensor, _ = torch.max(images, 1, keepdim=True)  # [8,2]
        # images = images/max_tensor  # 做一下归一化

        return images, labels
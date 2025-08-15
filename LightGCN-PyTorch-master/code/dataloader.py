"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import argparse
from collections import defaultdict

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")

        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems



    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader_prescription(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    # def __init__(self,config = world.config,path="../data/prescription"):
    #     # train or test
    #     cprint(f'loading [{path}]')
    #     self.split = config['A_split']
    #     self.folds = config['A_n_fold']
    #     self.mode_dict = {'train': 0, "test": 1}
    #     self.mode = self.mode_dict['train']
    #     self.n_user = 3396
    #     self.m_item = 2076
    #     train_file = path + '/train.txt'
    #     test_file = path + '/test.txt'
    #     self.path = path
    #     trainUniqueUsers, trainItem, trainUser = [], [], []
    #     testUniqueUsers, testItem, testUser = [], [], []
    #     self.traindataSize = 0
    #     self.testDataSize = 0
    #     print(train_file)
    #     with open(train_file) as f:
    #         for l in f.readlines():
    #             if len(l) > 0:
    #                 # print(l)
    #                 l = l.strip('\n').split(' ')
    #                 # 原代码（易出错）
    #                 # items = [int(i) for i in l[1:]]
    #                 # 修改为（带过滤）
    #                 items = [int(i) for i in l[1:] if i.strip() != '']
    #                 uid = int(l[0])
    #                 trainUniqueUsers.append(uid)
    #                 trainUser.extend([uid] * len(items))
    #                 trainItem.extend(items)
    #                 #  统计用户数 n_user 和物品数 m_item
    #                 # print(items)
    #                 self.m_item = max(self.m_item, max(items))
    #                 self.n_user = max(self.n_user, uid)
    #                 # print('self.n_user')
    #                 # print(self.n_user)
    #                 self.traindataSize += len(items)
    #     self.trainUniqueUsers = np.array(trainUniqueUsers)
    #     self.trainUser = np.array(trainUser)
    #     self.trainItem = np.array(trainItem)
    #
    #     with open(test_file) as f:
    #         for l in f.readlines():
    #             if len(l) > 0:
    #                 l = l.strip('\n').split(' ')
    #                 items = [int(i) for i in l[1:]]
    #                 uid = int(l[0])
    #                 testUniqueUsers.append(uid)
    #                 testUser.extend([uid] * len(items))
    #                 testItem.extend(items)
    #                 self.m_item = max(self.m_item, max(items))
    #                 self.n_user = max(self.n_user, uid)
    #                 self.testDataSize += len(items)
    #     # 我的编号就是从1开始所以无需+1
    #     # self.m_item += 1
    #     # self.n_user += 1
    #     # print(self.n_user)
    #     self.testUniqueUsers = np.array(testUniqueUsers)
    #     self.testUser = np.array(testUser)
    #     self.testItem = np.array(testItem)
    #
    #     self.Graph = None
    #     print(f"{self.trainDataSize} interactions for training")
    #     print(f"{self.testDataSize} interactions for testing")
    #     print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
    #
    #     # (users,items), bipartite graph  构建用户-物品二分图
    #     # 用 scipy.sparse.csr_matrix 存储稀疏交互矩阵
    #     # 矩阵形状：(n_user, m_item)，非零位置表示存在交互
    #     # print(f"max(self.trainUser):{max(self.trainUser)}")
    #     # print(f"self.n_user:{self.n_user}")
    #     # 如果ID从1开始
    #     self.trainUser = self.trainUser - 1  # 转换为0-based
    #     self.testUser = self.testUser - 1
    #     self.n_user = max(max(self.trainUser), max(self.testUser)) + 1
    #
    #     self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
    #                                   shape=(self.n_user, self.m_item))
    #     # self.users_D = np.array(self.UserItemNet.sum(axis=1))  # 用户度数
    #     # self.items_D = np.array(self.UserItemNet.sum(axis=0))  # 物品度数
    #     self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
    #     self.users_D[self.users_D == 0.] = 1
    #     self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
    #     self.items_D[self.items_D == 0.] = 1.
    #     # pre-calculate
    #     self._allPos = self.getUserPosItems(list(range(self.n_user)))
    #     self.__testDict = self.__build_test()
    #     print(f"{world.dataset} is ready to go")



    def __init__(self, config=world.config, path="../data"):
        """
        初始化数据集加载器

        参数:
            split_method:
                'leave-one-out' - 每个用户最后一个交互作为测试集
                'random' - 随机分割交互数据
        """
        # 获取命令行参数
        args = self.parse_args()
        self.split_method = config['split_method']

        # 初始化路径和基础配置
        self.path = path
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        # 固定维度设置
        if self.path == '../data/presF':
            self.n_user = 390  # 用户总数 (1-3396)
            self.m_item = 811  # 物品总数 (1-2076)
        elif self.path == '../data/prescription':
            self.n_user = 3396  # 用户总数 (1-3396)
            self.m_item = 2076  # 物品总数 (1-2076)




        # 初始化数据容器
        self.trainUser = np.array([])
        self.trainItem = np.array([])
        self.testUser = np.array([])
        self.testItem = np.array([])
        self.interaction_count = defaultdict(int)  # 用于统计交互频率
        # Load herb names from Excel file
        self.herb_names = self._load_herb_names()
        self.herb_cooccur = self._load_herb_cooccur()
        print(f"Loaded {len(self.herb_cooccur)} herb co-occurrence pairs")


        print('split method:',self.split_method)
        print('fold number:',self.folds)
        self._load_full_dataset()
        # 根据分割方法加载数据
        if self.split_method == 'random_split':
            self._load_random_split()
        elif self.split_method == 'leave_one_out':
            self._load_leave_one_out()
        else:
            raise ValueError(f"未知的分割方法: {self.split_method}")
        self._convert_ids_to_0_based()

        # 构建用户-物品交互矩阵
        self._build_interaction_matrix()

        # 打印数据集统计信息
        self._print_stats()

    def _load_herb_names(self):
        """Load herb names from Excel file"""
        import pandas as pd
        try:
            df = pd.read_excel('herbs_project.xlsx')
            herb_names = df.iloc[:, 0].tolist()  # Assuming herbs are in first column
            return herb_names
        except Exception as e:
            print(f"Failed to load herb names: {e}")
            return []

    @property
    def herb_name_to_id(self):
        """Create mapping from herb name to ID"""
        return {name: idx for idx, name in enumerate(self.herb_names)}

    def _convert_ids_to_0_based(self):
        """将所有ID转换为0-based索引"""
        # 用户ID转换
        self.trainUser = self.trainUser - 1
        self.testUser = self.testUser - 1

        # 物品ID转换
        self.trainItem = self.trainItem - 1
        self.testItem = self.testItem - 1

        # 更新交互计数字典的键
        new_interaction_count = {}
        for (disease_id, herb_id), count in self.interaction_count.items():
            new_key = (disease_id - 1, herb_id - 1)  # 转换为0-based
            new_interaction_count[new_key] = count
        self.interaction_count = new_interaction_count

        # 验证转换结果
        print(f"转换后最小用户ID: {min(self.trainUser)}")
        print(f"转换后最大用户ID: {max(self.trainUser)}")
        print(f"转换后最小物品ID: {min(self.trainItem)}")
        print(f"转换后最大物品ID: {max(self.trainItem)}")
        assert min(self.trainUser) >= 0, "训练集中有负的用户ID"
        assert min(self.testUser) >= 0, "测试集中有负的用户ID"
        assert min(self.trainItem) >= 0, "训练集中有负的物品ID"
        assert min(self.testItem) >= 0, "测试集中有负的物品ID"

    def parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--split_method', type=str, default='random',
                            choices=['random', 'leave-one-out'],
                            help='数据分割方法: random或leave-one-out')
        return parser.parse_args()

    def _load_full_dataset(self):
        """加载完整的数据集"""
        # full_data_file = os.path.join(self.path, 'symptom_herb_mapping.txt')
        # trainUser = []
        # trainItem = []
        # with open(full_data_file, 'r') as f:
        #     for line in f.readlines():
        #         if len(line) > 0:
        #             line = line.strip().split()
        #             disease_id = int(line[0])
        #             herb_ids = [int(i) for i in line[1:]]
        #             for herb_id in herb_ids:
        #                 trainUser.append(disease_id)
        #                 trainItem.append(herb_id)
        # self.trainUser = np.array(trainUser)
        # self.trainItem = np.array(trainItem)
        """加了交互次数统计"""
        full_data_file = os.path.join(self.path, 'disease_herb_interactions.txt')
        trainUser = []
        trainItem = []
        # 修改后代码（忽略频次）
        with open(full_data_file, 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip().split()
                    disease_id = int(line[0])
                    for herb_info in line[1:]:
                        herb_id = int(herb_info.split(':')[0])  # 只取物品ID
                        trainUser.append(disease_id)
                        trainItem.append(herb_id)
                        # 记录交互（计数为1）
                        self.interaction_count[(disease_id, herb_id)] = 1
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)


    # def _load_random_split(self):
    #     # 保证每个用户都有测试数据
    #     user_items = defaultdict(list)
    #     for uid, iid in zip(self.trainUser, self.trainItem):
    #         user_items[uid].append(iid)
    #
    #     train_pairs, test_pairs = [], []
    #     for uid, items in user_items.items():
    #         if len(items) >= 2:  # 至少2个交互才能分割
    #             np.random.shuffle(items)
    #             split_idx = max(1, int(0.8 * len(items)))  # 至少保留1个测试
    #             train_pairs.extend([(uid, iid) for iid in items[:split_idx]])
    #             test_pairs.extend([(uid, iid) for iid in items[split_idx:]])
    #
    #     # 添加统计信息
    #     print(f"RandomSplit Stats: {len(np.unique([x[0] for x in train_pairs]))} train users, "
    #           f"{len(np.unique([x[0] for x in test_pairs]))} test users")

    def _load_random_split(self):
        """改进的随机分割方法，保证每个用户在测试集中至少有一个样本"""
        # 收集所有用户的交互记录
        user_interactions = defaultdict(list)
        for uid, iid in zip(self.trainUser, self.trainItem):
            user_interactions[uid].append(iid)

        train_pairs = []
        test_pairs = []

        # 对每个用户进行独立分割
        for uid, items in user_interactions.items():
            if len(items) < 2:  # 少于2个交互的用户全部放入训练集
                train_pairs.extend([(uid, iid) for iid in items])
                continue

            # 随机打乱并分割
            np.random.shuffle(items)
            split_idx = max(1, int(0.8 * len(items)))  # 至少保留1个测试样本
            train_pairs.extend([(uid, iid) for iid in items[:split_idx]])
            test_pairs.extend([(uid, iid) for iid in items[split_idx:]])

        # 转换为numpy数组
        self.trainUser = np.array([x[0] for x in train_pairs])
        self.trainItem = np.array([x[1] for x in train_pairs])
        self.testUser = np.array([x[0] for x in test_pairs])
        self.testItem = np.array([x[1] for x in test_pairs])

        # 打印统计信息
        unique_train_users = len(np.unique(self.trainUser))
        unique_test_users = len(np.unique(self.testUser))
        print(f"Data Split Summary:")
        print(f"  Train Users: {unique_train_users} | Train Interactions: {len(train_pairs)}")
        print(f"  Test Users: {unique_test_users} | Test Interactions: {len(test_pairs)}")
        print(f"  Test Coverage: {unique_test_users / self.n_users * 100:.1f}%")

    def _load_leave_one_out(self):
        """留一法分割方法，每个用户最后一个交互作为测试集"""
        user_interactions = defaultdict(list)
        for uid, iid in zip(self.trainUser, self.trainItem):
            user_interactions[uid].append(iid)

        train_pairs = []
        test_pairs = []

        for uid, items in user_interactions.items():
            if len(items) < 1:
                continue
            test_item = items[-1]
            train_items = items[:-1]
            train_pairs.extend([(uid, iid) for iid in train_items])
            test_pairs.append((uid, test_item))

        self.trainUser = np.array([x[0] for x in train_pairs])
        self.trainItem = np.array([x[1] for x in train_pairs])
        self.testUser = np.array([x[0] for x in test_pairs])
        self.testItem = np.array([x[1] for x in test_pairs])

        unique_train_users = len(np.unique(self.trainUser))
        unique_test_users = len(np.unique(self.testUser))
        print(f"Data Split Summary:")
        print(f"  Train Users: {unique_train_users} | Train Interactions: {len(train_pairs)}")
        print(f"  Test Users: {unique_test_users} | Test Interactions: {len(test_pairs)}")
        print(f"  Test Coverage: {unique_test_users / self.n_users * 100:.1f}%")

    def _build_interaction_matrix(self):
        """构建用户-物品交互矩阵"""
        # 转换为0-based索引
        trainUser_0based = self.trainUser
        trainItem_0based = self.trainItem

        # 构建稀疏矩阵
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)),
             (trainUser_0based, trainItem_0based)),
            shape=(self.n_user, self.m_item)
        )

        # 计算度数
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # 预计算正样本
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.Graph = None

    def _print_stats(self):
        """打印数据集统计信息"""
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.testUniqueUsers = np.unique(self.testUser)
        self.traindataSize = len(self.trainUser)
        self.testDataSize = len(self.testUser)

        print(f"训练集交互数: {self.traindataSize}")
        print(f"测试集交互数: {self.testDataSize}")
        print(f"稀疏度: {(self.traindataSize + self.testDataSize) / self.n_user / self.m_item:.6f}")
        print(f"{world.dataset} 准备就绪 (用户数: {self.n_user}, 物品数: {self.m_item})")
        print(f"分割方法: {self.split_method}")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _load_herb_cooccur(self):
        """从方剂数据中提取草药共现关系（增加过滤和权重计算）"""
        import pandas as pd
        from collections import defaultdict

        # 读取方剂数据
        excel_path = os.path.join(self.path, 'pre_data_id.xlsx')
        df = pd.read_excel(excel_path, sheet_name='pre_data_id', header=None, engine='openpyxl')

        cooccur = defaultdict(int)
        herb_freq = defaultdict(int)  # 新增：草药频率统计

        # 第一遍：计算草药频率
        for _, row in df.iterrows():
            herbs = list(map(int, row[1].split()))
            valid_herbs = [h for h in herbs if 0 < h <= self.m_item]  # 仅保留此项过滤

            for i in range(len(valid_herbs)):
                for j in range(i + 1, len(valid_herbs)):
                    h1, h2 = sorted([valid_herbs[i], valid_herbs[j]])
                    cooccur[(h1, h2)] += 1  # 直接计数

            # 对数归一化代替PMI
            max_count = max(cooccur.values()) if cooccur else 1
            for pair in cooccur:
                cooccur[pair] = np.log(1 + cooccur[pair]) / np.log(1 + max_count)

            return cooccur  # 不再进行额外过滤

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                # 修改缓存文件名以包含共现信息
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_cooccur.npz')
                print("successfully loaded co-occurrence graph...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix with herb co-occurrence")
                s = time()
                # 1. 创建空的DOK矩阵
                n_nodes = self.n_users + self.m_items
                adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

                # 2. 添加自环（对角线设为1）
                for i in range(n_nodes):
                    adj_mat[i, i] = 1.0

                # 3. 添加带权重的用户-物品交互边
                max_count = max(self.interaction_count.values()) if self.interaction_count else 1
                for (user, item), count in self.interaction_count.items():
                    # 使用已经转换的0-based ID
                    u_idx = user
                    i_idx = item

                    # 应用对数归一化权重
                    weight = np.log(1 + count) / np.log(1 + max_count)

                    # 在二分图中设置权重（对称位置）
                    adj_mat[u_idx, self.n_users + i_idx] = weight
                    adj_mat[self.n_users + i_idx, u_idx] = weight

                # 4. 添加草药共现边
                max_cooccur = max(self.herb_cooccur.values()) if self.herb_cooccur else 1
                for (herb1, herb2), count in self.herb_cooccur.items():
                    # 转换为0-based索引
                    h1 = herb1 - 1
                    h2 = herb2 - 1

                    # 确保草药ID在有效范围内
                    if h1 < self.m_item and h2 < self.m_item:
                        # 应用对数归一化权重
                        weight = np.log(1 + count) / np.log(1 + max_cooccur)

                        # 在草药子图中设置权重（对称位置）
                        idx1 = self.n_users + h1
                        idx2 = self.n_users + h2
                        adj_mat[idx1, idx2] = weight
                        adj_mat[idx2, idx1] = weight

                # 5. 转换为LIL格式以便高效操作
                adj_mat = adj_mat.tolil()

                # 6. 对称归一化
                rowsum = np.array(adj_mat.sum(axis=1)).flatten()
                d_inv_sqrt = np.power(rowsum, -0.5)
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

                norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_cooccur.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph



    # def __build_test(self):
    #     """
    #     return:
    #         dict: {user: [items]}
    #     """
    #     test_data = {}
    #     for i, item in enumerate(self.testItem):
    #         user = self.testUser[i]
    #         if test_data.get(user):
    #             test_data[user].append(item)
    #         else:
    #             test_data[user] = [item]
    #     return test_data
    def __build_test(self):
        test_data = {}
        for user, item in zip(self.testUser, self.testItem):
            # 确保ID是0-based
            assert user >= 0, f"测试用户ID {user} 为负数"
            assert item >= 0, f"测试物品ID {item} 为负数"

            if user in test_data:
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    # def getUserPosItems(self, users):
    #     posItems = []
    #     for user in users:
    #         posItems.append(self.UserItemNet[user].nonzero()[1])
    #     return posItems

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            # 确保用户ID在有效范围内
            if user < 0 or user >= self.n_user:
                print(f"警告: 无效用户ID {user} (范围: 0-{self.n_user - 1})")
                posItems.append(np.array([], dtype=np.int64))
                continue

            items = self.UserItemNet[user].nonzero()[1]
            posItems.append(items)
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def getBaseGraph(self):
        """获取仅包含用户-物品交互的基础图"""
        if not hasattr(self, 'BaseGraph'):
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_base.npz')
                print("successfully loaded base graph...")
                norm_adj = pre_adj_mat
            except:
                print("generating base adjacency matrix (only user-item interactions)")
                s = time()
                n_nodes = self.n_users + self.m_items
                adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

                # 添加自环
                for i in range(n_nodes):
                    adj_mat[i, i] = 1.0

                # 只添加用户-物品交互边
                max_count = max(self.interaction_count.values()) if self.interaction_count else 1
                for (user, item), count in self.interaction_count.items():
                    u_idx = user
                    i_idx = item
                    weight = np.log(1 + count) / np.log(1 + max_count)
                    adj_mat[u_idx, self.n_users + i_idx] = weight
                    adj_mat[self.n_users + i_idx, u_idx] = weight

                adj_mat = adj_mat.tolil()

                # 对称归一化
                rowsum = np.array(adj_mat.sum(axis=1)).flatten()
                d_inv_sqrt = np.power(rowsum, -0.5)
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

                norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end - s}s, saved base norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_base.npz', norm_adj)

            if self.split == True:
                self.BaseGraph = self._split_A_hat(norm_adj)
            else:
                self.BaseGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.BaseGraph = self.BaseGraph.coalesce().to(world.device)
        return self.BaseGraph

    def getCooccurGraph(self):
        """获取仅包含草药共现关系的图"""
        if not hasattr(self, 'CooccurGraph'):
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_only_cooccur.npz')
                print("successfully loaded co-occurrence only graph...")
                norm_adj = pre_adj_mat
            except:
                print("generating co-occurrence only adjacency matrix")
                s = time()
                n_nodes = self.n_users + self.m_items
                adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

                # 添加自环
                for i in range(n_nodes):
                    adj_mat[i, i] = 1.0

                # 只添加草药共现边
                max_cooccur = max(self.herb_cooccur.values()) if self.herb_cooccur else 1
                for (herb1, herb2), count in self.herb_cooccur.items():
                    h1 = herb1 - 1
                    h2 = herb2 - 1
                    if h1 < self.m_item and h2 < self.m_item:
                        weight = np.log(1 + count) / np.log(1 + max_cooccur)
                        idx1 = self.n_users + h1
                        idx2 = self.n_users + h2
                        adj_mat[idx1, idx2] = weight
                        adj_mat[idx2, idx1] = weight

                adj_mat = adj_mat.tolil()

                # 对称归一化
                rowsum = np.array(adj_mat.sum(axis=1)).flatten()
                d_inv_sqrt = np.power(rowsum, -0.5)
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

                norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end - s}s, saved co-occurrence only norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_only_cooccur.npz', norm_adj)

            if self.split == True:
                self.CooccurGraph = self._split_A_hat(norm_adj)
            else:
                self.CooccurGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.CooccurGraph = self.CooccurGraph.coalesce().to(world.device)
        return self.CooccurGraph


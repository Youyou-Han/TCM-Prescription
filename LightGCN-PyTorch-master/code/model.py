"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
import os

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()    # 继承PyTorch的Module基类，调用父类nn.Module的初始化

    # 强制子类实现用户评分计算方法
    def getUsersRating(self, users):
        raise NotImplementedError  # 抽象方法：计算用户对物品的评分


#贝叶斯个性化排序损失函数
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()  # 继承BasicModel
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError  # 必须实现BPR损失计算
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users  # 从数据集获取用户数量
        self.num_items  = dataset.m_items  # 从数据集获取物品数量
        self.latent_dim = config['latent_dim_rec']  # 嵌入维度配置
        self.f = nn.Sigmoid()  # 评分标准化函数
        self.__init_weight()

    #从均值为0，标准差为0.1的正态分布采样初始值，生成user和item的初始embedding矩阵
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()  #将模型输出的原始分数映射到(0,1)区间，表示预测的点击/交互概率
        self.Graph = self.dataset.getSparseGraph()  # 获取用户-物品交互图的稀疏矩阵表示，构建用户-物品二部图的邻接矩阵
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()  # 获取稀疏矩阵尺寸
        index = x.indices().t()  # 获取非零元素的坐标
        values = x.values()  # 获取非零元素值
        # 生成随机掩码进行丢弃
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        # 构建新的稀疏矩阵
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])  # 合并用户和物品嵌入
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]  # 存储各层嵌入
        if self.config['dropout']:
            if self.training:  # 只在训练时dropout
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph

        # 多层图卷积传播
        for layer in range(self.n_layers):
            if self.A_split:  # 处理分割的邻接矩阵
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCN_llm(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN_llm, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 形状应为 [num_users, 768]
        # user_embed_init = np.load('D:/Program Files/code/LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/prescription/prescription_symptoms_embeddings.npy')  # 形状 [num_users, latent_dim]
        # item_embed_init = np.load('D:/Program Files/code/LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/prescription/prescription_herbs_embeddings.npy')  # 形状 [num_items, latent_dim]
        if self.config['pretrain'] == 1:
            if self.config['dataset'] == 'prescription':
                user_embed_init = np.load(
                    'D:/Program Files/code/LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/prescription/origin_symptoms_embeddings.npy')  # 形状 [num_users, latent_dim]
                item_embed_init = np.load(
                    'D:/Program Files/code/LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/prescription/origin_herbs_embeddings.npy')  # 形状 [num_items, latent_dim]
            elif self.config['dataset'] == 'presF':
                user_embed_init = np.load(
                    'D:/Program Files/code/LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/presF/prescription_symptoms_embeddings.npy')  # 形状 [num_users, latent_dim]
                item_embed_init = np.load(
                    'D:/Program Files/code/LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/presF/prescription_herbs_embeddings.npy')  # 形状 [num_items, latent_dim]
        # 使用PCA降维到目标维度
        # pca = PCA(n_components=self.latent_dim)
        # user_embed_reduced = pca.fit_transform(user_embed_init)
        # item_embed_reduced = pca.fit_transform(item_embed_init)

            if user_embed_init.shape[1] != self.latent_dim:
                pca = PCA(n_components=self.latent_dim)
                user_embed_reduced = pca.fit_transform(user_embed_init)
            else:
                user_embed_reduced = user_embed_init

            if item_embed_init.shape[1] != self.latent_dim:
                pca = PCA(n_components=self.latent_dim)
                item_embed_reduced = pca.fit_transform(item_embed_init)
            else:
                item_embed_reduced = item_embed_init

            self.embedding_user.weight.data.copy_(
                torch.from_numpy(user_embed_reduced).to(world.device))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(item_embed_reduced).to(world.device))
            print('use pretarined data')
        elif self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()  # 将模型输出的原始分数映射到(0,1)区间，表示预测的点击/交互概率
        self.Graph = self.dataset.getSparseGraph()  # 获取用户-物品交互图的稀疏矩阵表示，构建用户-物品二部图的邻接矩阵
        print(f"lgn is already to go(dropout:{self.config['dropout']})")


        # print("save_txt")

    # def _compute_edge_weight(self):
    #     """根据用户-物品交互频率计算边权重"""
    #     # 获取交互频率
    #     interaction_count = self.dataset.interaction_count
    #
    #     # 创建与嵌入维度匹配的权重向量
    #     n_nodes = self.num_users + self.num_items
    #     edge_weight = torch.ones(n_nodes, device=world.device)
    #
    #     # 如果需要基于交互频率设置权重，可以在这里扩展
    #     # 例如：根据用户与物品的交互频率设置节点权重
    #     if interaction_count:
    #         for (user_id, item_id), count in interaction_count.items():
    #             # 注意：这里需要根据实际数据格式调整索引
    #             user_idx = user_id - 1
    #             item_idx = item_id - 1
    #
    #             # 增加用户和物品节点的权重
    #             edge_weight[user_idx] += count
    #             edge_weight[self.num_users + item_idx] += count
    #
    #         # 归一化权重
    #         edge_weight = edge_weight / edge_weight.max()
    #
    #     return edge_weight

    def __dropout_x(self, x, keep_prob):
        size = x.size()  # 获取稀疏矩阵尺寸
        print('size:'+size)
        keep_prob = keep_prob / keep_prob.sum()
        index = x.indices().t()  # 获取非零元素的坐标
        values = x.values()  # 获取非零元素值
        # 生成随机掩码进行丢弃
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        # 构建新的稀疏矩阵
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        # # print(users_emb.size())
        # # print(items_emb.size())
        # all_emb = torch.cat([users_emb, items_emb])  # 合并用户和物品嵌入
        # # print(all_emb.shape)
        # #   torch.split(all_emb , [self.num_users, self.num_items])
        # embs = [all_emb]  # 存储各层嵌入
        # if self.config['dropout']:
        #     if self.training:  # 只在训练时dropout
        #         print("droping")
        #         g_droped = self.__dropout(self.keep_prob)
        #     else:
        #         g_droped = self.Graph
        # else:
        #     g_droped = self.Graph
        #
        # # 多层图卷积传播
        # for layer in range(self.n_layers):
        #     if self.A_split:  # 处理分割的邻接矩阵
        #         temp_emb = []
        #         for f in range(len(g_droped)):
        #             temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
        #         side_emb = torch.cat(temp_emb, dim=0)
        #         all_emb = side_emb
        #     else:
        #         # print(g_droped.shape)
        #         # print(g_droped.shape)
        #         all_emb = torch.sparse.mm(g_droped, all_emb)
        #     embs.append(all_emb)
        # embs = torch.stack(embs, dim=1)
        # # print(embs.size())
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])
        # return users, items
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        # 多层图卷积传播
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # 应用边权重：在图卷积操作中使用加权邻接矩阵
                all_emb = torch.sparse.mm(g_droped, all_emb)

            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class LightGCN_semantic(BasicModel):
    def __init__(self, config, dataset):
        super(LightGCN_semantic, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # 原始用户和物品嵌入
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # self.gate_user = nn.Linear(2 * self.latent_dim, 1)  # 用于用户嵌入融合
        # self.gate_item = nn.Linear(2 * self.latent_dim, 1)  # 用于物品嵌入融合

        # 语义嵌入（预训练语义向量）
        self.symptom_embedding = None  # 用户侧语义（如症状）
        self.herb_embedding = None  # 物品侧语义（如草药）
        # 深层融合的可学习权重（每一层单独设置权重）
        self.user_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers + 1)  # 包含初始层
        ])
        self.item_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers + 1)
        ])

        # 加载预训练语义向量
        if self.config['pretrain'] == 1:
            if self.config['dataset'] in ['prescription', 'presF']:
                try:
                    # 加载语义向量（路径根据实际情况修改）
                    if self.config['dataset'] == 'prescription':
                        symptom_emb = np.load('data/prescription/prescription_symptoms_embeddings_new.npy')
                        herb_emb = np.load('data/prescription/prescription_herbs_embeddings_new.npy')
                    else:
                        symptom_emb = np.load('data/presF/prescription_symptoms_embeddings_new.npy')
                        herb_emb = np.load('data/presF/prescription_herbs_embeddings_new.npy')

                    # 调整语义向量维度与模型一致
                    symptom_emb = self._adjust_embedding_dim(symptom_emb, self.latent_dim)
                    herb_emb = self._adjust_embedding_dim(herb_emb, self.latent_dim)

                    # 转换为可训练的嵌入层
                    self.symptom_embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(symptom_emb), freeze=False)  # 允许微调
                    self.herb_embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(herb_emb), freeze=False)
                    print("Successfully loaded and initialized semantic embeddings for deep fusion")
                except Exception as e:
                    print(f"Failed to load semantic embeddings: {e}")

                # 初始化原始嵌入
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

            self.f = nn.Sigmoid()
            self.Graph = self.dataset.getSparseGraph()
            print(f"Deep fusion model initialized with {self.n_layers} layers")



    def _adjust_embedding_dim(self, embedding, target_dim):
        """调整嵌入维度（与原逻辑一致）"""
        if embedding.shape[1] > target_dim:
            from sklearn.decomposition import PCA
            return PCA(n_components=target_dim).fit_transform(embedding)
        elif embedding.shape[1] < target_dim:
            new_emb = np.zeros((embedding.shape[0], target_dim))
            new_emb[:, :embedding.shape[1]] = embedding
            return new_emb
        return embedding

    def computer(self):
        """深层融合的图传播：每一层都融合语义信息"""
        # 1. 初始化基础嵌入和语义嵌入
        users_emb = self.embedding_user.weight  # 结构嵌入（用户）
        items_emb = self.embedding_item.weight  # 结构嵌入（物品）

        # 获取语义嵌入（如果存在）
        symptom_emb = self.symptom_embedding(torch.arange(self.num_users).to(world.device)) if self.symptom_embedding else None
        herb_emb = self.herb_embedding(torch.arange(self.num_items).to(world.device)) if self.herb_embedding else None

        # 2. 初始层融合（第0层）
        if symptom_emb is not None:
            # 第0层用户融合：结构嵌入 * 权重 + 语义嵌入 * (1-权重)
            alpha = torch.sigmoid(self.user_layer_weights[0])
            users_emb = alpha * users_emb + (1 - alpha) * symptom_emb
        if herb_emb is not None:
            # 第0层物品融合
            beta = torch.sigmoid(self.item_layer_weights[0])
            items_emb = beta * items_emb + (1 - beta) * herb_emb

        # 合并初始嵌入并记录
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]  # 存储每一层的融合结果

        # 3. 多层图卷积传播（从第1层到第n_layers层）
        g_droped = self.__dropout(self.keep_prob) if self.config['dropout'] and self.training else self.Graph

        for layer in range(1, self.n_layers + 1):  # 从第1层开始
            # 图卷积传播（结构信息更新）
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)

            # 分割出当前层的用户和物品嵌入
            current_users = all_emb[:self.num_users]
            current_items = all_emb[self.num_users:]

            # 深层语义融合（当前层）
            if symptom_emb is not None:
                alpha = torch.sigmoid(self.user_layer_weights[layer])  # 该层的用户融合权重
                current_users = alpha * current_users + (1 - alpha) * symptom_emb
            if herb_emb is not None:
                beta = torch.sigmoid(self.item_layer_weights[layer])  # 该层的物品融合权重
                current_items = beta * current_items + (1 - beta) * herb_emb

            # 合并更新后的嵌入
            all_emb = torch.cat([current_users, current_items])
            embs.append(all_emb)

        # 4. 聚合所有层的结果
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)  # 平均各层嵌入
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def __dropout_x(self, x, keep_prob):
        size = x.size()  # 获取稀疏矩阵尺寸
        print('size:'+size)
        keep_prob = keep_prob / keep_prob.sum()
        index = x.indices().t()  # 获取非零元素的坐标
        values = x.values()  # 获取非零元素值
        # 生成随机掩码进行丢弃
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        # 构建新的稀疏矩阵
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def getUsersRating(self, users):
        """获取用户评分（使用融合语义后的嵌入）"""
        all_users, all_items = self.computer()  # 这里会自动调用融合语义的computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        if self.symptom_embedding is not None:
            symptom_emb = self.symptom_embedding(users)
            herb_pos_emb = self.herb_embedding(pos_items)
            herb_neg_emb = self.herb_embedding(neg_items)
            # 只返回前6个值，忽略语义嵌入
            return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
        else:
            return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        """简化版forward，使用computer()的融合结果"""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCN_Cooccur(BasicModel):
    def __init__(self, config, dataset):
        super(LightGCN_Cooccur, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
        # 添加分层门控机制
        self.gate_networks = nn.ModuleList()
        for _ in range(self.n_layers + 1):  # +1 包含初始层
            gate_network = nn.Sequential(
                nn.Linear(self.latent_dim * 2, 32),  # 输入：结构嵌入+语义嵌入
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.gate_networks.append(gate_network)

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # 添加共现边权重参数
        self.cooccur_weight = nn.Parameter(torch.tensor(0.5))  # 共现边的整体权重
        self.layer_cooccur_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers)  # 每层的共现权重
        ])

        # 原始用户和物品嵌入
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # 语义嵌入（预训练语义向量）
        self.symptom_embedding = None  # 用户侧语义（如症状）
        self.herb_embedding = None  # 物品侧语义（如草药）

        # 深层融合的可学习权重（每一层单独设置权重）
        self.user_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers + 1)  # 包含初始层
        ])
        self.item_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers + 1)
        ])

        # 加载预训练语义向量
        if self.config['pretrain'] == 1:
            if self.config['dataset'] in ['prescription', 'presF']:
                try:
                    # 加载语义向量
                    if self.config['dataset'] == 'prescription':
                        symptom_emb = np.load('../data/prescription/prescription_symptoms_embeddings_new.npy')
                        herb_emb = np.load('../data/prescription/prescription_herbs_embeddings_new.npy')
                    else:
                        symptom_emb = np.load('../data/presF/prescription_symptoms_embeddings_new.npy')
                        herb_emb = np.load('../data/presF/prescription_herbs_embeddings_new.npy')

                    # 调整语义向量维度与模型一致
                    symptom_emb = self._adjust_embedding_dim(symptom_emb, self.latent_dim)
                    herb_emb = self._adjust_embedding_dim(herb_emb, self.latent_dim)

                    # 转换为可训练的嵌入层
                    self.symptom_embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(symptom_emb), freeze=False)
                    self.herb_embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(herb_emb), freeze=False)
                    print("Successfully loaded and initialized semantic embeddings for deep fusion")
                except Exception as e:
                    print(f"Failed to load semantic embeddings: {e}")

            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

            self.f = nn.Sigmoid()
            # 分别获取基础交互图和共现图
            self.BaseGraph = self.dataset.getBaseGraph()
            self.CooccurGraph = self.dataset.getCooccurGraph()
            print(f"Deep fusion model initialized with {self.n_layers} layers and weighted co-occurrence")


    def computer(self):
        """深层融合的图传播：每一层都融合语义信息和动态调整共现权重"""
        # 1. 初始化基础嵌入和语义嵌入
        users_emb = self.embedding_user.weight  # 结构嵌入（用户）
        items_emb = self.embedding_item.weight  # 结构嵌入（物品）

        # 获取语义嵌入（如果存在）
        symptom_emb = self.symptom_embedding(
            torch.arange(self.num_users).to(world.device)) if self.symptom_embedding else None
        herb_emb = self.herb_embedding(
            torch.arange(self.num_items).to(world.device)) if self.herb_embedding else None
        # 2. 初始层融合
        if symptom_emb is not None:
            # 使用门控网络决定融合比例
            gate_input = torch.cat([users_emb, symptom_emb], dim=1)
            alpha = self.gate_networks[0](gate_input)
            users_emb = alpha * users_emb + (1 - alpha) * symptom_emb

        if herb_emb is not None:
            gate_input = torch.cat([items_emb, herb_emb], dim=1)
            beta = self.gate_networks[0](gate_input)
            items_emb = beta * items_emb + (1 - beta) * herb_emb
        # 合并初始嵌入并记录
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]  # 存储每一层的融合结果

        # 3. 分层解耦传播
        for layer in range(1, self.n_layers + 1):
            # 独立计算基础图和共现图的传播
            base_emb = torch.sparse.mm(self.BaseGraph, all_emb)
            cooccur_emb = torch.sparse.mm(self.CooccurGraph, all_emb)

            # 分割用户和物品节点
            base_users, base_items = torch.split(base_emb, [self.num_users, self.num_items])
            cooccur_users, cooccur_items = torch.split(cooccur_emb, [self.num_users, self.num_items])

            # 节点级门控融合（仅物品节点）
            if herb_emb is not None:
                # 物品节点：结合结构嵌入和语义嵌入计算门控值
                gate_input = torch.cat([base_items, herb_emb], dim=1)
                item_gates = self.gate_networks[layer](gate_input)

                # 应用门控融合
                fused_items = item_gates * base_items + (1 - item_gates) * cooccur_items
            else:
                fused_items = base_items

            # 用户节点直接使用基础图传播结果
            fused_users = base_users

            # 合并融合结果
            fused_emb = torch.cat([fused_users, fused_items])

            # 添加当前层嵌入
            embs.append(fused_emb)
            all_emb = fused_emb  # 更新为下一层输入

        # 4. 聚合所有层的结果
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def __dropout(self, keep_prob, graph):
        if self.A_split:
            drop_graph = []
            for g in graph:
                drop_graph.append(self.__dropout_x(g, keep_prob))
        else:
            drop_graph = self.__dropout_x(graph, keep_prob)
        return drop_graph

    # 其他方法保持不变...
    def _adjust_embedding_dim(self, embedding, target_dim):
        if embedding.shape[1] > target_dim:
            from sklearn.decomposition import PCA
            return PCA(n_components=target_dim).fit_transform(embedding)
        elif embedding.shape[1] < target_dim:
            new_emb = np.zeros((embedding.shape[0], target_dim))
            new_emb[:, :embedding.shape[1]] = embedding
            return new_emb
        return embedding

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        if self.symptom_embedding is not None:
            symptom_emb = self.symptom_embedding(users)
            herb_pos_emb = self.herb_embedding(pos_items)
            herb_neg_emb = self.herb_embedding(neg_items)
            return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
        else:
            return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        # 添加共现权重的正则化
        reg_loss += 0.01 * (self.cooccur_weight.norm(2).pow(2) +
                            sum(w.norm(2).pow(2) for w in self.layer_cooccur_weights))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # === 修改正则化策略 ===
        # 增加共现权重的稀疏约束
        cooccur_reg = 0.1 * (torch.sigmoid(self.cooccur_weight) - 0.5).abs().sum()
        layer_reg = 0.05 * sum((torch.sigmoid(w) - 0.5).abs().sum() for w in self.layer_cooccur_weights)

        reg_loss += cooccur_reg + layer_reg
        # === 修改结束 ===

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCN_Symptom(BasicModel):  # 重命名为 LightGCN_Symptom
    def __init__(self, config, dataset):
        """
        在LightGCN_Cooccur基础上添加疾病和草药特征融合
        保留所有原始功能并添加症状特征融合
        """
        super(LightGCN_Symptom, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()
        # 添加分层门控机制 - 保留原始代码
        self.gate_networks = nn.ModuleList()
        for _ in range(self.n_layers + 1):  # +1 包含初始层
            gate_network = nn.Sequential(
                nn.Linear(self.latent_dim * 2, 32),  # 输入：结构嵌入+语义嵌入
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.gate_networks.append(gate_network)

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # 添加共现边权重参数 - 保留原始代码
        self.cooccur_weight = nn.Parameter(torch.tensor(0.5))  # 共现边的整体权重
        self.layer_cooccur_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers)  # 每层的共现权重
        ])

        # 原始用户和物品嵌入 - 保留原始代码
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # === 新增：症状和草药特征 ===
        # 症状特征维度（用户侧）
        self.symptom_dim = self.config.get('symptom_dim', 768)
        # 草药特征维度（物品侧）
        self.herb_dim = self.config.get('herb_dim', 768)

        # 加载预训练症状和草药向量
        self.symptom_embedding = None  # 用户侧症状特征
        self.herb_embedding = None  # 物品侧草药特征

        # 深层融合的可学习权重（每一层单独设置权重）- 保留原始代码
        self.user_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers + 1)  # 包含初始层
        ])
        self.item_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.n_layers + 1)
        ])

        # 加载预训练语义向量 - 保留原始代码并增强
        if self.config['pretrain'] == 1:
            if self.config['dataset'] in ['prescription', 'presF']:
                try:
                    # 加载语义向量
                    if self.config['dataset'] == 'prescription':
                        symptom_emb = np.load('../data/prescription/prescription_symptoms_embeddings_new.npy')
                        herb_emb = np.load('../data/prescription/prescription_herbs_embeddings_new.npy')
                    else:
                        symptom_emb = np.load('../data/presF/prescription_symptoms_embeddings_new.npy')
                        herb_emb = np.load('../data/presF/prescription_herbs_embeddings_new.npy')


                    # 调整语义向量维度与模型一致
                    symptom_emb = self._adjust_embedding_dim(symptom_emb, self.latent_dim)
                    herb_emb = self._adjust_embedding_dim(herb_emb, self.latent_dim)

                    # 转换为可训练的嵌入层
                    self.symptom_embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(symptom_emb), freeze=False)
                    self.herb_embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(herb_emb), freeze=False)
                    print("Successfully loaded and initialized symptom and herb embeddings for deep fusion")
                except Exception as e:
                    print(f"Failed to load semantic embeddings: {e}")
                    # 回退方案：使用随机初始化
                    self.symptom_embedding = nn.Embedding(
                        num_embeddings=self.num_users,
                        embedding_dim=self.latent_dim
                    )
                    self.herb_embedding = nn.Embedding(
                        num_embeddings=self.num_items,
                        embedding_dim=self.latent_dim
                    )
                    nn.init.normal_(self.symptom_embedding.weight, std=0.1)
                    nn.init.normal_(self.herb_embedding.weight, std=0.1)
                    print("Initialized random symptom and herb embeddings")
            else:
                # 对于其他数据集，使用随机初始化
                self.symptom_embedding = nn.Embedding(
                    num_embeddings=self.num_users,
                    embedding_dim=self.latent_dim
                )
                self.herb_embedding = nn.Embedding(
                    num_embeddings=self.num_items,
                    embedding_dim=self.latent_dim
                )
                nn.init.normal_(self.symptom_embedding.weight, std=0.1)
                nn.init.normal_(self.herb_embedding.weight, std=0.1)
                print("Initialized random symptom and herb embeddings for non-prescription dataset")
        else:
            # 不使用预训练时，使用随机初始化
            self.symptom_embedding = nn.Embedding(
                num_embeddings=self.num_users,
                embedding_dim=self.latent_dim
            )
            self.herb_embedding = nn.Embedding(
                num_embeddings=self.num_items,
                embedding_dim=self.latent_dim
            )
            nn.init.normal_(self.symptom_embedding.weight, std=0.1)
            nn.init.normal_(self.herb_embedding.weight, std=0.1)
            print("Initialized random symptom and herb embeddings (pretrain=0)")

        # 初始化结构嵌入 - 保留原始代码
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()
        # 分别获取基础交互图和共现图 - 保留原始代码
        self.BaseGraph = self.dataset.getBaseGraph()
        self.CooccurGraph = self.dataset.getCooccurGraph()
        print(f"Deep fusion model initialized with {self.n_layers} layers and weighted co-occurrence")

        # === 新增：症状特征投影层 ===
        self.symptom_proj = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(512, self.latent_dim)
        )
        nn.init.xavier_uniform_(self.symptom_proj[0].weight)
        nn.init.zeros_(self.symptom_proj[0].bias)
        nn.init.xavier_uniform_(self.symptom_proj[3].weight)
        nn.init.zeros_(self.symptom_proj[3].bias)

        # === 新增：草药特征投影层 ===
        self.herb_proj = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(512, self.latent_dim)
        )
        nn.init.xavier_uniform_(self.herb_proj[0].weight)
        nn.init.zeros_(self.herb_proj[0].bias)
        nn.init.xavier_uniform_(self.herb_proj[3].weight)
        nn.init.zeros_(self.herb_proj[3].bias)

        # === 新增：门控融合机制 ===
        self.symptom_fusion_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.herb_fusion_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.symptom_fusion_gate.weight)
        nn.init.zeros_(self.symptom_fusion_gate.bias)
        nn.init.xavier_uniform_(self.herb_fusion_gate.weight)
        nn.init.zeros_(self.herb_fusion_gate.bias)

    def _enhance_embeddings(self, user_ids, item_ids):
        """
        增强嵌入：融合ID嵌入和症状/草药特征
        """
        # 原始ID嵌入
        user_emb = self.embedding_user(user_ids)
        item_emb = self.embedding_item(item_ids)

        # 获取症状特征
        symptom_emb = self.symptom_embedding(user_ids)
        # 投影症状特征
        projected_symptom = self.symptom_proj(symptom_emb)

        # 门控融合症状特征
        user_combined = torch.cat([user_emb, projected_symptom], dim=1)
        user_gate = torch.sigmoid(self.symptom_fusion_gate(user_combined))
        enhanced_user_emb = user_gate * user_emb + (1 - user_gate) * projected_symptom

        # 获取草药特征
        herb_emb = self.herb_embedding(item_ids)
        # 投影草药特征
        projected_herb = self.herb_proj(herb_emb)

        # 门控融合草药特征
        item_combined = torch.cat([item_emb, projected_herb], dim=1)
        item_gate = torch.sigmoid(self.herb_fusion_gate(item_combined))
        enhanced_item_emb = item_gate * item_emb + (1 - item_gate) * projected_herb

        return enhanced_user_emb, enhanced_item_emb

    def computer(self):
        """深层融合的图传播：每一层都融合语义信息和动态调整共现权重"""
        # 1. 初始化基础嵌入和语义嵌入
        users_emb = self.embedding_user.weight  # 结构嵌入（用户）
        items_emb = self.embedding_item.weight  # 结构嵌入（物品）

        # === 修改点1：使用增强嵌入 ===
        # 获取所有用户ID和物品ID
        all_user_ids = torch.arange(self.num_users).to(world.device)
        all_item_ids = torch.arange(self.num_items).to(world.device)

        # 获取增强嵌入
        users_emb, items_emb = self._enhance_embeddings(all_user_ids, all_item_ids)

        # 2. 初始层融合 - 保留原始代码
        symptom_emb = self.symptom_embedding(
            torch.arange(self.num_users).to(world.device)) if self.symptom_embedding else None
        herb_emb = self.herb_embedding(
            torch.arange(self.num_items).to(world.device)) if self.herb_embedding else None

        if symptom_emb is not None:
            # 使用门控网络决定融合比例
            gate_input = torch.cat([users_emb, symptom_emb], dim=1)
            alpha = self.gate_networks[0](gate_input)
            users_emb = alpha * users_emb + (1 - alpha) * symptom_emb

        if herb_emb is not None:
            gate_input = torch.cat([items_emb, herb_emb], dim=1)
            beta = self.gate_networks[0](gate_input)
            items_emb = beta * items_emb + (1 - beta) * herb_emb

        # 合并初始嵌入并记录 - 保留原始代码
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]  # 存储每一层的融合结果

        # 3. 分层解耦传播 - 保留原始代码
        for layer in range(1, self.n_layers + 1):
            # 独立计算基础图和共现图的传播
            base_emb = torch.sparse.mm(self.BaseGraph, all_emb)
            cooccur_emb = torch.sparse.mm(self.CooccurGraph, all_emb)

            # 分割用户和物品节点
            base_users, base_items = torch.split(base_emb, [self.num_users, self.num_items])
            cooccur_users, cooccur_items = torch.split(cooccur_emb, [self.num_users, self.num_items])

            # 节点级门控融合（仅物品节点）- 保留原始代码
            if herb_emb is not None:
                # 物品节点：结合结构嵌入和语义嵌入计算门控值
                gate_input = torch.cat([base_items, herb_emb], dim=1)
                item_gates = self.gate_networks[layer](gate_input)

                # 应用门控融合
                fused_items = item_gates * base_items + (1 - item_gates) * cooccur_items
            else:
                fused_items = base_items

            # 用户节点直接使用基础图传播结果 - 保留原始代码
            fused_users = base_users

            # 合并融合结果 - 保留原始代码
            fused_emb = torch.cat([fused_users, fused_items])

            # 添加当前层嵌入 - 保留原始代码
            embs.append(fused_emb)
            all_emb = fused_emb  # 更新为下一层输入

        # 4. 聚合所有层的结果 - 保留原始代码
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def __dropout(self, keep_prob, graph):
        # 保留原始代码
        if self.A_split:
            drop_graph = []
            for g in graph:
                drop_graph.append(self.__dropout_x(g, keep_prob))
        else:
            drop_graph = self.__dropout_x(graph, keep_prob)
        return drop_graph

    def __dropout_x(self, x, keep_prob):
        # 保留原始代码
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def _adjust_embedding_dim(self, embedding, target_dim):
        # 保留原始代码
        if embedding.shape[1] > target_dim:
            from sklearn.decomposition import PCA
            return PCA(n_components=target_dim).fit_transform(embedding)
        elif embedding.shape[1] < target_dim:
            new_emb = np.zeros((embedding.shape[0], target_dim))
            new_emb[:, :embedding.shape[1]] = embedding
            return new_emb
        return embedding

    def getUsersRating(self, users):
        # 保留原始代码
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        # 保留原始代码
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        # === 修改点2：使用增强嵌入 ===
        enhanced_user_emb, _ = self._enhance_embeddings(users, pos_items)
        _, enhanced_pos_emb = self._enhance_embeddings(users, pos_items)
        _, enhanced_neg_emb = self._enhance_embeddings(users, neg_items)

        # 返回增强嵌入和原始嵌入
        return (users_emb, pos_emb, neg_emb,
                enhanced_user_emb, enhanced_pos_emb, enhanced_neg_emb,
                users_emb_ego, pos_emb_ego, neg_emb_ego)

    def bpr_loss(self, users, pos, neg):
        # 保留原始代码并修改嵌入获取
        (users_emb, pos_emb, neg_emb,
         enhanced_user_emb, enhanced_pos_emb, enhanced_neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        # 添加共现权重的正则化
        reg_loss += 0.01 * (self.cooccur_weight.norm(2).pow(2) +
                            sum(w.norm(2).pow(2) for w in self.layer_cooccur_weights))

        # 使用增强嵌入计算分数
        pos_scores = torch.sum(enhanced_user_emb * enhanced_pos_emb, dim=1)
        neg_scores = torch.sum(enhanced_user_emb * enhanced_neg_emb, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # 增加共现权重的稀疏约束
        cooccur_reg = 0.1 * (torch.sigmoid(self.cooccur_weight) - 0.5).abs().sum()
        layer_reg = 0.05 * sum((torch.sigmoid(w) - 0.5).abs().sum() for w in self.layer_cooccur_weights)

        reg_loss += cooccur_reg + layer_reg

        return loss, reg_loss

    def forward(self, users, items):
        # 保留原始代码
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

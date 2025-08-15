'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from parse import parse_args

from sklearn.metrics import roc_auc_score
args = parse_args()

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
# def test_one_batch(X):
#     sorted_items = X[0].numpy()
#     groundTrue = X[1]
#     r = utils.getLabel(groundTrue, sorted_items)
#     pre, recall, ndcg = [], [], []
#     for k in world.topks:
#         ret = utils.RecallPrecision_ATk(groundTrue, r, k)
#         pre.append(ret['precision'])
#         recall.append(ret['recall'])
#         ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
#     return {'recall':np.array(recall),
#             'precision':np.array(pre),
#             'ndcg':np.array(ndcg)}
        
            
# def Test(dataset, Recmodel, epoch, w=None, multicore=0):
#     u_batch_size = world.config['test_u_batch_size']
#     dataset: utils.BasicDataset
#     testDict: dict = dataset.testDict
#     Recmodel: model.LightGCN
#     # eval mode with no dropout
#     Recmodel = Recmodel.eval()
#     max_K = max(world.topks)
#     if multicore == 1:
#         pool = multiprocessing.Pool(CORES)
#     results = {'precision': np.zeros(len(world.topks)),
#                'recall': np.zeros(len(world.topks)),
#                'ndcg': np.zeros(len(world.topks))}
#     with torch.no_grad():
#         users = list(testDict.keys())
#         try:
#             assert u_batch_size <= len(users) / 10
#         except AssertionError:
#             print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#         users_list = []
#         rating_list = []
#         groundTrue_list = []
#         # auc_record = []
#         # ratings = []
#         total_batch = len(users) // u_batch_size + 1
#         for batch_users in utils.minibatch(users, batch_size=u_batch_size):
#             allPos = dataset.getUserPosItems(batch_users)
#             groundTrue = [testDict[u] for u in batch_users]
#             batch_users_gpu = torch.Tensor(batch_users).long()
#             batch_users_gpu = batch_users_gpu.to(world.device)
#
#             rating = Recmodel.getUsersRating(batch_users_gpu)
#             #rating = rating.cpu()
#             exclude_index = []
#             exclude_items = []
#             for range_i, items in enumerate(allPos):
#                 exclude_index.extend([range_i] * len(items))
#                 exclude_items.extend(items)
#             rating[exclude_index, exclude_items] = -(1<<10)
#             _, rating_K = torch.topk(rating, k=max_K)
#             rating = rating.cpu().numpy()
#             # aucs = [
#             #         utils.AUC(rating[i],
#             #                   dataset,
#             #                   test_data) for i, test_data in enumerate(groundTrue)
#             #     ]
#             # auc_record.extend(aucs)
#             del rating
#             users_list.append(batch_users)
#             rating_list.append(rating_K.cpu())
#             groundTrue_list.append(groundTrue)
#         assert total_batch == len(users_list)
#         X = zip(rating_list, groundTrue_list)
#         if multicore == 1:
#             pre_results = pool.map(test_one_batch, X)
#         else:
#             pre_results = []
#             for x in X:
#                 pre_results.append(test_one_batch(x))
#         scale = float(u_batch_size/len(users))
#         for result in pre_results:
#             results['recall'] += result['recall']
#             results['precision'] += result['precision']
#             results['ndcg'] += result['ndcg']
#         results['recall'] /= float(len(users))
#         results['precision'] /= float(len(users))
#         results['ndcg'] /= float(len(users))
#         # results['auc'] = np.mean(auc_record)
#         if world.tensorboard:
#             w.add_scalars(f'Test/Recall@{world.topks}',
#                           {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
#             w.add_scalars(f'Test/Precision@{world.topks}',
#                           {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
#             w.add_scalars(f'Test/NDCG@{world.topks}',
#                           {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
#         if multicore == 1:
#             pool.close()
#         print(results)
#         return results



# def test_one_batch(X):
#     if args.split_methods == 'leave_one_out':
#         sorted_items = X[0].numpy()
#         groundTrue = X[1]
#         r = utils.getLabel(groundTrue, sorted_items)
#         pre, recall, ndcg = [], [], []
#         for k in world.topks:
#             ret = utils.RecallPrecision_ATk(groundTrue, r, k)
#             pre.append(ret['precision'])
#             recall.append(ret['recall'])
#             ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
#         return {'recall': np.array(recall),
#                 'precision': np.array(pre),
#                 'ndcg': np.array(ndcg)}
#     elif args.split_methods == 'random_split':
#         sorted_items = X[0].numpy()  # [batch_size, max_K]
#         groundTrue = X[1]  # [batch_size, list]
#
#         batch_size = len(groundTrue)
#         recall = np.zeros(len(world.topks))
#         precision = np.zeros(len(world.topks))
#         ndcg = np.zeros(len(world.topks))
#
#         for i, k in enumerate(world.topks):
#             # 生成命中矩阵（兼容原NDCG函数）
#             r = utils.getLabel(groundTrue, sorted_items[:, :k])
#
#             for u in range(batch_size):
#                 # Recall@K
#                 hits = np.sum(r[u])
#                 recall[i] += hits / len(groundTrue[u])
#
#                 # Precision@K
#                 precision[i] += hits / k
#
#                 # NDCG@K（选择任一方案）
#                 # ndcg[i] += NDCGatK_r(groundTrue, sorted_items[:, :k], k)  # 方案2
#                 # 或保持原逻辑：
#                 ndcg[i] += np.sum(utils.NDCGatK_r([groundTrue[u]], [r[u]], k))  # 方案1
#
#             # 平均到batch内用户
#             recall[i] /= batch_size
#             precision[i] /= batch_size
#             ndcg[i] /= batch_size
#
#         return {'recall': recall, 'precision': precision, 'ndcg': ndcg}

# def test_one_batch(X):
#     """统一处理两种分割方式的评估逻辑"""
#     sorted_items = X[0].numpy()  # 模型预测的排序列表 [batch_size, k]
#     groundTrue = X[1]  # 真实测试物品 [batch_size, test_items]
#
#     batch_size = len(groundTrue)
#     topks = world.topks
#     max_k = max(topks)
#
#     # 初始化结果容器
#     result = {
#         'precision': np.zeros(len(topks)),
#         'recall': np.zeros(len(topks)),
#         'ndcg': np.zeros(len(topks))
#     }
#
#     # 计算每个K的指标
#     for i, k in enumerate(topks):
#         precision_sum = 0
#         recall_sum = 0
#         ndcg_sum = 0
#
#         for u in range(batch_size):
#             # 获取当前用户的预测和真实物品
#             pred = set(sorted_items[u, :k])
#             true = set(groundTrue[u])
#
#             # 计算命中数
#             hits = len(pred & true)
#
#             # Precision@K
#             precision_sum += hits / k
#
#             # Recall@K (关键修改：分母使用测试集物品数)
#             recall_sum += hits / len(true) if len(true) > 0 else 0
#
#             # NDCG@K
#             relevance = np.isin(sorted_items[u, :k], groundTrue[u]).astype(int)
#             dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
#             ideal_relevance = np.sort(relevance)[::-1]
#             # idcg = np.sum(ideal_relevance / np.log2(np.arange(2, min(k, len(true)) + 2)))
#             # 计算IDCG - 修改后的部分
#             L = min(k, len(true))  # 实际相关物品数
#             if L > 0:
#                 ideal_relevance = np.ones(L)
#                 discount = np.log2(np.arange(2, L + 2))
#                 idcg = np.sum(ideal_relevance / discount)
#             else:
#                 idcg = 0.0
#             ndcg_sum += dcg / (idcg + 1e-8)  # 避免除以零
#
#         # 计算batch内平均
#         result['precision'][i] = precision_sum / batch_size
#         result['recall'][i] = recall_sum / batch_size
#         result['ndcg'][i] = ndcg_sum / batch_size
#
#     return result

def test_one_batch(X, dataset=None):
    sorted_items = X[0].numpy()  # 模型预测的Top-K草药 [batch_size, K]
    groundTrue = X[1]  # 真实草药列表 [batch_size, variable_length]
    batch_users = X[2]  # 用户ID [batch_size]

    batch_size = len(groundTrue)
    topks = world.topks  # 例如 [5, 10, 20]

    result = {
        'precision': np.zeros(len(topks)),
        'recall': np.zeros(len(topks)),
        'f1': np.zeros(len(topks)),
        'ndcg': np.zeros(len(topks))
    }

    for i, k in enumerate(topks):
        precision_sum = recall_sum = f1_sum = ndcg_sum = 0
        valid_users = 0

        for u in range(batch_size):
            true_herbs = set(groundTrue[u])
            pred_herbs = set(sorted_items[u, :k])

            if len(true_herbs) == 0:
                continue  # 跳过无真实交互的用户

            # 计算指标
            hits = len(pred_herbs & true_herbs)
            precision = hits / k
            recall = hits / len(true_herbs) if len(true_herbs) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # NDCG计算（兼容PresRecRF的版本）
            relevance = np.isin(sorted_items[u, :k], list(true_herbs)).astype(float)
            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
            idcg = np.sum(np.ones(min(len(true_herbs), k)) / np.log2(np.arange(2, min(len(true_herbs), k) + 2)))
            ndcg = dcg / (idcg + 1e-8)

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            ndcg_sum += ndcg
            valid_users += 1

        # 批次内平均
        if valid_users > 0:
            result['precision'][i] = precision_sum / valid_users
            result['recall'][i] = recall_sum / valid_users
            result['f1'][i] = f1_sum / valid_users
            result['ndcg'][i] = ndcg_sum / valid_users

    return result

# def Test(dataset, Recmodel, epoch, w=None, multicore=0):
#     """统一的测试流程，兼容两种分割方式"""
#     dataset: utils.BasicDataset
#     Recmodel: model.LightGCN
#     testDict = dataset.testDict
#
#     # 确保使用eval模式
#     Recmodel = Recmodel.eval()
#     max_k = max(world.topks)
#
#     # 过滤掉测试集中没有正样本的用户
#     valid_users = [u for u in testDict if len(testDict[u]) > 0]
#     if len(valid_users) < len(testDict):
#         print(f"Warning: {len(testDict) - len(valid_users)} users have no test items")
#
#     # 初始化结果容器
#     final_results = {
#         'precision': np.zeros(len(world.topks)),
#         'recall': np.zeros(len(world.topks)),
#         'ndcg': np.zeros(len(world.topks))
#     }
#
#     # 分批处理
#     with torch.no_grad():
#         user_batches = list(utils.minibatch(
#             valid_users,
#             batch_size=world.config['test_u_batch_size']
#         ))
#
#         for batch_users in tqdm(user_batches, desc="Testing"):
#             # 获取测试集正样本
#             groundTrue = [testDict[u] for u in batch_users]
#
#             # 转换为tensor并送入设备
#             batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
#
#             # 获取预测评分
#             all_ratings = Recmodel.getUsersRating(batch_users_gpu)
#
#             # 排除训练集物品 - 修复索引越界问题
#             trainPos = dataset.getUserPosItems(batch_users)
#             for i, items in enumerate(trainPos):
#                 # 只处理存在的物品索引
#                 valid_items = [item for item in items if item < dataset.m_items]
#                 if valid_items:
#                     all_ratings[i, valid_items] = -float('inf')
#
#             # 获取Top-K推荐
#             _, topk_indices = torch.topk(all_ratings, k=max_k)
#
#             # 计算指标
#             batch_results = test_one_batch((
#                 topk_indices.cpu(),
#                 groundTrue
#             ))
#
#             # 累加结果
#             for metric in final_results:
#                 final_results[metric] += batch_results[metric] * len(batch_users)
#
#     # 最终平均
#     for metric in final_results:
#         final_results[metric] /= len(valid_users)
#
#     # 打印结果
#     print("\nTest Results:")
#     print(f"  Users Evaluated: {len(valid_users)}/{len(testDict)}")
#     for i, k in enumerate(world.topks):
#         print(f"  Top-{k}: "
#               f"Precision={final_results['precision'][i]:.4f}, "
#               f"Recall={final_results['recall'][i]:.4f}, "
#               f"NDCG={final_results['ndcg'][i]:.4f}")
#
#     # TensorBoard记录
#     if world.tensorboard and w is not None:
#         for i, k in enumerate(world.topks):
#             w.add_scalar(f'Test/Precision@{k}', final_results['precision'][i], epoch)
#             w.add_scalar(f'Test/Recall@{k}', final_results['recall'][i], epoch)
#             w.add_scalar(f'Test/NDCG@{k}', final_results['ndcg'][i], epoch)
#
#     return final_results

def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    Recmodel.eval()
    max_k = max(world.topks)

    # 过滤无测试交互的用户
    valid_users = [u for u in dataset.testDict if len(dataset.testDict[u]) > 0]
    print(f"有效测试用户: {len(valid_users)}/{dataset.n_users}")

    final_results = {
        'precision': np.zeros(len(world.topks)),
        'recall': np.zeros(len(world.topks)),
        'f1': np.zeros(len(world.topks)),
        'ndcg': np.zeros(len(world.topks))
    }

    with torch.no_grad():
        for batch_users in utils.minibatch(valid_users, batch_size=world.config['test_u_batch_size']):
            # 获取测试集正样本
            groundTrue = [dataset.testDict[u] for u in batch_users]

            # 模型预测
            batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
            all_ratings = Recmodel.getUsersRating(batch_users_gpu)

            # 排除训练集物品（防止数据泄漏）
            trainPos = dataset.getUserPosItems(batch_users)
            for i, items in enumerate(trainPos):
                all_ratings[i, items] = -float('inf')

            # 获取Top-K推荐
            _, topk_indices = torch.topk(all_ratings, k=max_k)

            # 计算指标
            batch_results = test_one_batch((
                topk_indices.cpu(),
                groundTrue,
                batch_users
            ))

            # 加权累加（按用户数）
            for metric in final_results:
                final_results[metric] += batch_results[metric] * len(batch_users)

        # 全局平均
        for metric in final_results:
            final_results[metric] /= len(valid_users)

    # 打印结果（格式与PresRecRF完全一致）
    print("\n测试结果 (PresRecRF风格评估):")
    for i, k in enumerate(world.topks):
        print(f"  Top-{k}: "
              f"Precision={final_results['precision'][i]:.4f}, "
              f"Recall={final_results['recall'][i]:.4f}, "
              f"F1={final_results['f1'][i]:.4f}, "
              f"NDCG={final_results['ndcg'][i]:.4f}")

    return final_results

# def Test(dataset, Recmodel, epoch, w=None, multicore=0):
#     if args.split_methods == 'leave_one_out':
#         u_batch_size = world.config['test_u_batch_size']
#         dataset: utils.BasicDataset
#         testDict: dict = dataset.testDict
#         Recmodel: model.LightGCN
#         # eval mode with no dropout
#         Recmodel = Recmodel.eval()
#         max_K = max(world.topks)
#         if multicore == 1:
#             pool = multiprocessing.Pool(CORES)
#         results = {'precision': np.zeros(len(world.topks)),
#                    'recall': np.zeros(len(world.topks)),
#                    'ndcg': np.zeros(len(world.topks))}
#         with torch.no_grad():
#             users = list(testDict.keys())
#             try:
#                 assert u_batch_size <= len(users) / 10
#             except AssertionError:
#                 print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#             users_list = []
#             rating_list = []
#             groundTrue_list = []
#             # auc_record = []
#             # ratings = []
#             total_batch = len(users) // u_batch_size + 1
#             for batch_users in utils.minibatch(users, batch_size=u_batch_size):
#                 allPos = dataset.getUserPosItems(batch_users)
#                 groundTrue = [testDict[u] for u in batch_users]
#                 batch_users_gpu = torch.Tensor(batch_users).long()
#                 batch_users_gpu = batch_users_gpu.to(world.device)
#
#                 rating = Recmodel.getUsersRating(batch_users_gpu)
#                 # rating = rating.cpu()
#                 exclude_index = []
#                 exclude_items = []
#                 for range_i, items in enumerate(allPos):
#                     exclude_index.extend([range_i] * len(items))
#                     exclude_items.extend(items)
#                 rating[exclude_index, exclude_items] = -(1 << 10)
#                 _, rating_K = torch.topk(rating, k=max_K)
#                 rating = rating.cpu().numpy()
#                 # aucs = [
#                 #         utils.AUC(rating[i],
#                 #                   dataset,
#                 #                   test_data) for i, test_data in enumerate(groundTrue)
#                 #     ]
#                 # auc_record.extend(aucs)
#                 del rating
#                 users_list.append(batch_users)
#                 rating_list.append(rating_K.cpu())
#                 groundTrue_list.append(groundTrue)
#             assert total_batch == len(users_list)
#             X = zip(rating_list, groundTrue_list)
#             if multicore == 1:
#                 pre_results = pool.map(test_one_batch, X)
#             else:
#                 pre_results = []
#                 for x in X:
#                     pre_results.append(test_one_batch(x))
#             scale = float(u_batch_size / len(users))
#             for result in pre_results:
#                 results['recall'] += result['recall']
#                 results['precision'] += result['precision']
#                 results['ndcg'] += result['ndcg']
#             results['recall'] /= float(len(users))
#             results['precision'] /= float(len(users))
#             results['ndcg'] /= float(len(users))
#             # results['auc'] = np.mean(auc_record)
#             if world.tensorboard:
#                 w.add_scalars(f'Test/Recall@{world.topks}',
#                               {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
#                 w.add_scalars(f'Test/Precision@{world.topks}',
#                               {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
#                 w.add_scalars(f'Test/NDCG@{world.topks}',
#                               {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
#             if multicore == 1:
#                 pool.close()
#             print(results)
#             return results
#     elif args.split_methods == 'random_split':
#         u_batch_size = world.config['test_u_batch_size']
#         dataset: utils.BasicDataset
#         testDict: dict = dataset.testDict  # {user: [test_item1, test_item2, ...]}
#         Recmodel: model.LightGCN
#         Recmodel = Recmodel.eval()
#         max_K = max(world.topks)
#
#         if multicore == 1:
#             pool = multiprocessing.Pool(CORES)
#
#         results = {
#             'precision': np.zeros(len(world.topks)),
#             'recall': np.zeros(len(world.topks)),
#             'ndcg': np.zeros(len(world.topks))
#         }
#
#         with torch.no_grad():
#             users = list(testDict.keys())
#             try:
#                 assert u_batch_size <= len(users) / 10
#             except AssertionError:
#                 print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
#
#             users_list = []
#             rating_list = []
#             groundTrue_list = []
#
#             # 分批处理用户
#             for batch_users in utils.minibatch(users, batch_size=u_batch_size):
#                 # 获取每个用户的所有正样本（训练集+测试集）
#                 allPos = dataset.getUserPosItems(batch_users)
#                 # 获取测试集正样本（可能多个）
#                 groundTrue = [testDict[u] for u in batch_users]
#
#                 # 将用户ID转为Tensor并送入设备
#                 batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
#
#                 # 获取用户对所有物品的评分
#                 rating = Recmodel.getUsersRating(batch_users_gpu)
#
#                 # 排除训练集中的物品（避免重复推荐）
#                 exclude_index = []
#                 exclude_items = []
#                 for i, items in enumerate(allPos):
#                     exclude_index.extend([i] * len(items))
#                     exclude_items.extend(items)
#                 rating[exclude_index, exclude_items] = -(1 << 10)  # 用极小值掩码
#
#                 # 获取Top-K推荐物品
#                 _, rating_K = torch.topk(rating, k=max_K)
#
#                 # 释放显存
#                 del rating
#                 torch.cuda.empty_cache()
#
#                 # 保存结果用于后续指标计算
#                 users_list.append(batch_users)
#                 rating_list.append(rating_K.cpu())
#                 groundTrue_list.append(groundTrue)
#
#             # 计算指标（多进程/单进程）
#             X = zip(rating_list, groundTrue_list)
#             if multicore == 1:
#                 pre_results = pool.map(test_one_batch, X)
#             else:
#                 pre_results = [test_one_batch(x) for x in X]
#
#             # 汇总结果
#             for result in pre_results:
#                 results['recall'] += result['recall']
#                 results['precision'] += result['precision']
#                 results['ndcg'] += result['ndcg']
#
#             # 平均指标
#             results['recall'] /= len(users)
#             results['precision'] /= len(users)
#             results['ndcg'] /= len(users)
#
#             # 记录到TensorBoard
#             if world.tensorboard:
#                 w.add_scalars('Test/Recall@{world.topks}',
#                               {str(k): results['recall'][i] for i, k in enumerate(world.topks)}, epoch)
#                 w.add_scalars('Test/Precision@{world.topks}',
#                               {str(k): results['precision'][i] for i, k in enumerate(world.topks)}, epoch)
#                 w.add_scalars('Test/NDCG@{world.topks}',
#                               {str(k): results['ndcg'][i] for i, k in enumerate(world.topks)}, epoch)
#
#             if multicore == 1:
#                 pool.close()
#
#             print(f"Test Results Epoch {epoch}: {results}")
#             return results





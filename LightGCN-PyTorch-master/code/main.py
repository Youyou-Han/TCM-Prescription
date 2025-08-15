import world  # 导入自定义的world模块，通常包含全局配置参数
import utils  # 导入自定义的utils模块，包含工具函数
from world import cprint
import torch  # 导入PyTorch深度学习框架
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import Procedure
from os.path import join
import json
import pandas as pd
from pathlib import Path
# ==============================
# 设置随机种子（保证实验可复现性）
utils.set_seed(world.seed)
print(">>SEED:", world.seed)   # 打印当前使用的随机种子
# ==============================
# 导入register模块和其中的dataset对象
import register
from register import dataset

# 初始化模型
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

# 定义损失函数，BPRLoss: 贝叶斯个性化排序损失（Bayesian Personalized Ranking），常用于推荐系统
bpr = utils.BPRLoss(Recmodel, world.config)

# 获取权重文件名
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# -------------------------
# 新增：保存配置和词表
export_dir = join(world.FILE_PATH, "export")
Path(export_dir).mkdir(parents=True, exist_ok=True)

# 读取症状和草药的词表
SYM_XLSX = "../data/prescription/prescription_symptoms.xlsx"
HERB_XLSX = "../data/prescription/prescription_herbs.xlsx"
# 修改：使用openpyxl引擎读取xlsx文件
try:
    sym_names = pd.read_excel(SYM_XLSX, header=None, engine='openpyxl')[0].astype(str).str.strip().tolist()
    herb_names = pd.read_excel(HERB_XLSX, header=None, engine='openpyxl')[0].astype(str).str.strip().tolist()
    print(f"Loaded {len(sym_names)} symptom names")
    print(f"Loaded {len(herb_names)} herb names")
except Exception as e:
    print(f"Error loading vocab files: {e}")
    # 创建空列表防止后续崩溃
    sym_names = []
    herb_names = []
print(sym_names)
print(herb_names)

# 存词表
with open(join(export_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump({
        "symptoms": sym_names,  # index→name
        "herbs": herb_names
    }, f, ensure_ascii=False, indent=2)

# 存训练配置
with open(join(export_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "world_config": world.config,
        "args": {
            "dataset": world.dataset,
            "model": world.model_name,
            "epochs": world.TRAIN_epochs,
            "topks": world.topks,
            "seed": world.seed
        }
    }, f, ensure_ascii=False, indent=2)

# -------------------------
# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# -------------------------
# 开始训练
Neg_k = 1
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # 每10轮进行一次验证
        if epoch % 10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

        # 训练过程
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')

        # 每轮训练后保存权重
        torch.save(Recmodel.state_dict(), weight_file)

        # 判断是否是最优模型，可以保存最优模型（可根据验证损失来判断）
        # 如果有验证集损失或指标，可基于此条件来保存
        # 此处可以保存最佳模型的代码示例：
        if epoch == world.TRAIN_epochs - 1:  # 这里可以改成基于验证集最优的保存
            torch.save(Recmodel.state_dict(), join(export_dir, f'best_model_{world.model_name}_{epoch+1}.pt'))

finally:
    # 训练完成后关闭 tensorboard
    if world.tensorboard:
        w.close()

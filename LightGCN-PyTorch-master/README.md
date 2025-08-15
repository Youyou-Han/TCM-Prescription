## LightGCN-PyTorch（中医方剂推荐与剂量生成）

面向中医处方推荐的 LightGCN 实现，并结合本地 LLM（DeepSeek-R1-Distill-Qwen-1.5B）对推荐草药逐味生成建议剂量，仅供研究与工程验证使用。

### 功能特性

- **LightGCN 训练与评估**：支持 BPRLoss，周期性评测与 TensorBoard 可视化。
- **多数据集支持**：`prescription`（默认）、`presF`，以及经典数据集占位（gowalla/yelp/amazon-book/lastfm）。
- **多模型变体**：`mf`、`lgn`、`lightgcn_llm`、`lightgcn_semantic`、`lightgcn_cooccur`、`lightgcn_symptom`。
- **推理与剂量生成**：基于训练好的表示检索草药，调用本地 LLM 为每味草药生成建议剂量，并带有安全裁剪逻辑。
- **导出词表与配置**：训练时自动导出 `vocab.json` 与配置到 `code/checkpoints/export/`。

## 安装步骤

以下以 Windows/PowerShell 为例，推荐使用虚拟环境。

```bash
# 1) 克隆本项目（或直接复制现有目录）
cd "D:\Program Files\code"

# 2) 建议使用 Python 3.8~3.10（与较旧的 PyTorch 1.4.0 兼容）
python -m venv .venv
.\.venv\Scripts\activate

# 3) 安装基础依赖（训练所需）
cd .\LightGCN-PyTorch-master\
pip install -r requirements.txt

# 4) 额外依赖（推理/读表所需）
pip install openpyxl transformers accelerate safetensors

# 如需 TensorBoard（可选）
pip install tensorboard
```

提示：`requirements.txt` 使用较旧版本（如 `torch==1.4.0`、`pandas==0.24.2`）。若本地环境较新，可自行升级依赖，但需自行适配可能的 API 变化。

## 使用示例

### 训练

为保证相对路径正确，建议在 `code` 目录运行：

```bash
cd .\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code
python main.py ^
  --dataset prescription ^
  --model lightgcn_cooccur ^
  --epochs 400 ^
  --tensorboard 1 ^
  --recdim 256 ^
  --layer 2 ^
  --lr 5e-4 ^
  --decay 1e-4 ^
  --bpr_batch 2048 ^
  --topks "[10]" ^
  --seed 2020
```

训练过程中会：

- 每 10 个 epoch 在验证集评测一次。
- 自动将当前权重保存到 `code/checkpoints/`，并在最后导出最优权重到 `code/checkpoints/export/`（含 `vocab.json` 与 `config.json`）。

### 推理（草药 Top-K 推荐 + 逐味剂量生成）

1) 先确保已经完成一次训练并生成：

- `code/checkpoints/export/vocab.json`
- 一个可用的模型权重（默认示例名形如 `best_model_lightgcn_cooccur_400.pt`）

2) 打开并编辑 `code/predict.py` 顶部的常量路径（使用你本机实际路径）：

```python
GCN_WEIGHT_PATH = r"D:\\...\\code\\checkpoints\\export\\best_model_lightgcn_cooccur_400.pt"
LOCAL_MODEL_PATH = r"D:\\...\\models\\deepseek_R1"
```

3) 在 `code` 目录下以脚本方式调用：

```bash
cd .\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code
python -c "from predict import predict; print(predict('疮疹,腹泻', 10))"
```

或在 Python 中以编程方式使用：

```python
# 需在 code 目录内执行，或自行将该目录加入 sys.path
from predict import predict

res = predict("疮疹,腹泻", top_k=10)
print(res["input_symptoms"], res["topk"])         # 基本信息
for h in res["herbs"][:5]:                         # 查看前 5 条
    print(h["ID"], h["name"], h["value"], h["dosage"])
```

说明：若本地 LLM 未能成功加载，系统将回退到保守的安全剂量中值策略。

## 配置说明

### 关键命令行参数（节选，自 `code/parse.py`）

```text
--dataset         默认 prescription，可选 [lastfm, gowalla, yelp2018, amazon-book, prescription, presF]
--model           默认 lightgcn_cooccur，可选 [mf, lgn, lightgcn_llm, lightgcn_semantic, lightgcn_cooccur, lightgcn_symptom]
--epochs          默认 400
--bpr_batch       默认 2048（BPR 训练 batch 大小）
--recdim          默认 256（嵌入维度）
--layer           默认 2（LightGCN 层数）
--lr              默认 5e-4（学习率）
--decay           默认 1e-4（L2 正则）
--dropout         默认 0（是否使用 dropout）
--keepprob        默认 0.6（保留概率，配合 dropout）
--a_fold          默认 100（大稀疏图分块 fold 数）
--testbatch       默认 100（测试时用户 batch）
--multicore       默认 0（测试阶段是否多进程）
--pretrain        默认 1（是否使用预训练）
--topks           默认 "[10]"（评测 @k）
--tensorboard     默认 1（是否启用 TensorBoard）
--split_methods   默认 random_split，可选 [leave_one_out, random_split]
--comment         默认 "lgn"（日志注释）
--load            默认 0（是否从已有权重继续训练）
--seed            默认 2020（随机种子）
--path            默认 ./checkpoints（保存路径）
```

### 数据与路径

- 数据目录：`LightGCN-PyTorch-master/LightGCN-PyTorch-master/data/prescription/`
- 关键文件（仓库已提供示例）：
  - `train.txt`、`test.txt`：训练/测试交互，行格式一般为 `user item1 item2 ...`（基于 1 起始 ID，内部会转换为 0 起始）。
  - `prescription_symptoms.xlsx`、`prescription_herbs.xlsx`：症状与草药词表（训练时读取并导出到 `vocab.json`）。
  - 其他 `.npz`、`txt` 文件：用于建图、统计或扩展特征（如共现矩阵等）。

注意：由于代码中相对路径多以 `code` 目录为基准（例如 `../data/...`），建议在 `code` 目录内运行训练与推理命令。

### LLM 推理

- 模型目录结构示例：`models/deepseek_R1/`（已随仓库提供示例权重与配置文件）。
- 需安装：`transformers`、`accelerate`、`safetensors`。
- 若使用 GPU，可在 `code/predict.py` 中将 `USE_GPU_HALF = True` 保持开启以节省显存（需要支持 FP16 的 GPU 与驱动）。

## 技术栈

- 语言：Python
- 训练/表示学习：PyTorch（LightGCN）、NumPy、SciPy
- 数据处理：Pandas、openpyxl（读取 .xlsx）
- 可视化：TensorBoard（通过 `SummaryWriter`）
- 推理（可选）：Transformers（本地 LLM）

## 目录结构

```text
LightGCN-PyTorch-master/
├─ LightGCN-PyTorch-master/
│  ├─ code/
│  │  ├─ main.py            # 训练入口
│  │  ├─ world.py           # 全局配置/路径/设备管理
│  │  ├─ parse.py           # CLI 参数
│  │  ├─ model.py           # 模型与变体
│  │  ├─ dataloader.py      # 数据加载与划分
│  │  ├─ Procedure.py       # 训练与评测过程
│  │  ├─ utils.py           # 采样、BPRLoss、工具函数
│  │  ├─ register.py        # 数据集与模型路由
│  │  ├─ predict.py         # 推理与逐味剂量生成
│  │  ├─ runs/              # TensorBoard 日志
│  │  └─ checkpoints/       # 训练权重与导出（含 export/）
│  ├─ data/
│  │  └─ prescription/      # 数据集（示例）
│  └─ models/
│     └─ deepseek_R1/       # 本地 LLM（示例）
└─ requirements.txt         # 训练依赖
```

## 贡献指南

欢迎提交 Issue 与 Pull Request！建议流程：

1. Fork 本仓库并新建分支（例如 `feat/xxx` 或 `fix/xxx`）。
2. 本地开发与测试：
   - 确保在 `code` 目录可完成训练/推理的基本跑通。
   - 对新增功能补充必要的注释与文档。
3. 提交 PR 时请描述：动机、主要改动点、兼容性影响、测试要点。

## 许可证信息

- 本仓库根目录暂未提供统一开源许可证。若需开源发布，请在根目录添加 `LICENSE` 并在 README 中声明。
- 目录 `models/deepseek_R1/` 中包含其模型的独立许可证与文件，使用时请遵循对应条款。

---

如需在你的环境中适配新版依赖或替换本地 LLM，欢迎在 Issue 中交流反馈。



# -*- coding: utf-8 -*-
"""
predict.py — LightGCN 检索 + 本地 LLM（DeepSeek-R1-Distill-Qwen-1.5B）逐味生成剂量
⚠ 本文件仅用于研究与工程调试；所有剂量输出需人工复核，不可用作医疗处方。
"""

import os
import re
import json
import torch
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Tuple
from os.path import join, exists

from transformers import AutoTokenizer, AutoModelForCausalLM

import world
from world import cprint
import register
from register import dataset
from torch.nn.functional import cosine_similarity
import networkx as nx
from os.path import join
import pandas as pd
# =========================
# 路径与基础配置（按需修改）
# =========================

# LightGCN 导出的权重（保持你的路径）
GCN_WEIGHT_PATH = r"D:\Program Files\code\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code\checkpoints\export\best_model_lightgcn_cooccur_400.pt"

# 本地 LLM：DeepSeek-R1-Distill-Qwen-1.5B 所在文件夹（按你的命名）
LOCAL_MODEL_PATH = r"D:\Program Files\code\LightGCN-PyTorch-master\LightGCN-PyTorch-master\models\deepseek_R1"

# 外部剂量区间文件（任选其一，存在即优先使用）
HERB_RANGE_TXT = r"D:\Program Files\code\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code\herb_dosage_ranges.txt"   # 形如 SAFE_DOSAGE_RANGES = {...}
HERB_RANGE_JSON = r"D:\Program Files\code\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code\herb_dosage_ranges.json"  # 形如 {"甘草":{"min":2,"max":10},...}

# 逐味生成：最多生成的 token（数字+小数点足够）
GEN_MAX_NEW_TOKENS_SINGLE = 6
USE_GPU_HALF = True  # 有 GPU 时是否半精度


# =========================
# 基线安全剂量范围（单位：克）— 若未找到外部文件则使用
# =========================
BASELINE_SAFE_RANGES: Dict[str, Tuple[float, float]] = {
    # —— 剧毒/矿物/贵重等（仅研究用示例）——
    "砒霜": (0.002, 0.004),
    "雄黄": (0.05, 0.1),
    "朱砂": (0.1, 0.5),
    "斑蝥": (0.03, 0.06),
    "蟾酥": (0.015, 0.03),
    "巴豆": (0.1, 0.3),
    "天仙子": (0.06, 0.6),
    "生半夏": (3, 9),
    "生南星": (3, 9),
    "洋金花": (0.3, 0.6),
    "雷公藤": (10, 25),
    "细辛": (1, 3),

    # —— 有毒 ——
    "甘遂": (0.5, 1.5),
    "狼毒": (0.9, 2.4),
    "商陆": (3, 9),
    "千金子": (1, 2),
    "牵牛子": (3, 6),
    "罂粟壳": (3, 6),
    "水蛭": (1.5, 3),
    "蜈蚣": (3, 5),
    "全蝎": (3, 6),

    # —— 常见非毒或小毒（仅示例）——
    "苦杏仁": (3, 10),
    "白果": (5, 10),
    "半夏": (3, 9),
    "天南星": (3, 9),
    "艾叶": (3, 9),
    "甘草": (2, 10),
    "当归": (6, 12),
    "黄芪": (9, 30),
    "人参": (3, 9),
    "大黄": (3, 15),
    "黄连": (2, 5),
}

# —— 补全本轮 10 味（仅研究用示例，需人工复核）——
BASELINE_SAFE_RANGES.update({
    "重楼": (3.0, 6.0),
    "使君子": (3.0, 10.0),
    "铅霜": (0.1, 0.3),
    "走石": (0.2, 1.0),
    "蛇黄": (0.05, 0.1),   # 同义词映射到雄黄见下方
    "天竺黄": (3.0, 6.0),
    "麝香": (0.03, 0.1),
    "胡黄连": (1.0, 3.0),
    "熊胆": (0.05, 0.2),
    # 朱砂 已在基线
})

GLOBAL_MIN, GLOBAL_MAX = 3.0, 15.0  # 全局兜底范围


def _to_half_step(x: float) -> float:
    """量化到 0.5g，并保持 1 位小数显示（如 10.0、8.5）"""
    return float(f"{round(float(x) * 2) / 2:.1f}")

def load_safe_ranges() -> Dict[str, Tuple[float, float]]:
    # 1) TXT: Python 字典文本
    if exists(HERB_RANGE_TXT):
        try:
            with open(HERB_RANGE_TXT, "r", encoding="utf-8") as f:
                ns = {}
                exec(f.read(), ns)
                d = ns.get("SAFE_DOSAGE_RANGES", {})
                d2 = {}
                for k, v in d.items():
                    try:
                        lo, hi = float(v[0]), float(v[1])
                        if hi >= lo > 0:
                            d2[str(k).strip()] = (lo, hi)
                    except Exception:
                        pass
                if d2:
                    print(f"[RANGE] 使用外部 TXT，共 {len(d2)} 条。")
                    return d2
        except Exception as e:
            print(f"[RANGE] 读取 TXT 失败：{e}")

    # 2) JSON: {"药名":{"min":x,"max":y}}
    if exists(HERB_RANGE_JSON):
        try:
            with open(HERB_RANGE_JSON, "r", encoding="utf-8") as f:
                j = json.load(f)
            d2 = {}
            for k, v in j.items():
                try:
                    lo, hi = float(v["min"]), float(v["max"])
                    if hi >= lo > 0:
                        d2[str(k).strip()] = (lo, hi)
                except Exception:
                    pass
            if d2:
                print(f"[RANGE] 使用外部 JSON，共 {len(d2)} 条。")
                return d2
        except Exception as e:
            print(f"[RANGE] 读取 JSON 失败：{e}")

    print(f"[RANGE] 使用内置基线，共 {len(BASELINE_SAFE_RANGES)} 条；其余走全局兜底 {GLOBAL_MIN}-{GLOBAL_MAX}g。")
    return dict(BASELINE_SAFE_RANGES)


SAFE_DOSAGE_RANGES = load_safe_ranges()

# =========================
# LLM 加载（DeepSeek-R1-Distill-Qwen-1.5B）
# =========================
_LLM_MODEL = None
_LLM_TOKENIZER = None
_MODEL_LOADED = False

def load_local_llm():
    """加载 DeepSeek-R1-Distill-Qwen-1.5B（AutoModelForCausalLM）"""
    global _LLM_MODEL, _LLM_TOKENIZER, _MODEL_LOADED
    if _MODEL_LOADED:
        return _LLM_MODEL, _LLM_TOKENIZER
    _MODEL_LOADED = True

    if not exists(LOCAL_MODEL_PATH):
        print(f"[LLM] 错误: 模型目录不存在: {LOCAL_MODEL_PATH}")
        return None, None

    print(f"[LLM] 从本地目录加载: {LOCAL_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if torch.cuda.is_available() and USE_GPU_HALF:
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
            )
            print("[LLM] 使用GPU + FP16")
        else:
            print("[LLM] 使用CPU")
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True).float()
            model.to("cpu")

        model.eval()
        _LLM_MODEL, _LLM_TOKENIZER = model, tokenizer
        print("[LLM] 模型加载成功")
        return model, tokenizer
    except Exception as e:
        print(f"[LLM] 模型加载失败: {e}")
        return None, None


def _single_herb_prompt(symptoms: str, herb: str, lo: float, hi: float) -> str:
    return (
        "你是中医方药剂量推断助手。请在给定剂量范围内，依据辨证与配伍原则选择**最合理**的单味剂量。"
        "你必须综合考虑：症状→证型→治法→方药配伍→本味药在方中的角色（君/臣/佐/使）、药性（四气五味、归经）、"
        "药对与常见组方（如四君子汤、四物汤、清热解毒方等），以及常见禁忌与十八反/十九畏。"
        "若症状信息不足，遵循“中量或略偏向量”的保守策略，并根据本味药是否可能担任君/臣/佐/使微调："
        "君≈1.0×基准量，臣≈0.8×，佐≈0.6×，使≈0.4×（最后仍需落在给定范围）。"
        "如处方中含与本味药相反相畏者，应相应下调或贴近下限；如协同药对明显（例如甘草-白芍/黄芩/大枣等），可适度上调。"
        "默认成年非孕非儿科，无严重肝肾功能异常。注意：你**不能**超出给定范围。"
        "\n\n"
        f"【症状（原始）】{symptoms}\n"
        f"【当前评估之单味草药】{herb}\n"
        f"【允许的剂量范围(克)】{lo}-{hi}\n"
        "请直接给出**唯一的阿拉伯数字**（克，保留1位小数），不要单位、不要任何解释或标点、不要空格，"
        "第一字符必须是0-9。例如：6.5"
    )

def _digit_token_ids(tokenizer):
    """获取 ASCII 数字 0-9 的 token id（仅收单 token 的映射）。"""
    ids = []
    for ch in "0123456789":
        toks = tokenizer.encode(ch, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return list(sorted(set(ids)))

def _dot_token_ids(tokenizer):
    """仅允许 ASCII '.' 的 token id；显式排除全角 '．'。"""
    ids = []
    toks = tokenizer.encode(".", add_special_tokens=False)
    if len(toks) == 1:
        ids.append(toks[0])
    return list(sorted(set(ids)))

def _digit_prefix_fn_builder(tokenizer, prompt_len: int):
    """位置感知的前缀约束：
       - 第1个生成 token：只能是 0-9
       - 后续 token：只能是 0-9 或 ASCII '.' （不允许全角点/负号）
    """
    digit_ids = _digit_token_ids(tokenizer)
    dot_ids = _dot_token_ids(tokenizer)

    def _fn(batch_id, input_ids):
        gen_len = input_ids.shape[0] - prompt_len  # 已生成 token 数（不含提示）
        if gen_len <= 0:
            return digit_ids
        else:
            return digit_ids + dot_ids

    return _fn

def _gen_number_for_herb(model, tokenizer, prompt: str, max_new_tokens: int = GEN_MAX_NEW_TOKENS_SINGLE) -> float:
    # 用 chat 模板更稳：系统+用户
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "只输出一个数字（克），保留1位小数，不要单位与解释。"},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.bool)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    prompt_len = inputs["input_ids"].shape[1]
    prefix_fn = _digit_prefix_fn_builder(tokenizer, prompt_len)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # ← 改成 True
        temperature=0.7,  # 可取 0.6~0.9
        top_p=0.9,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    m = re.search(r"\d+(?:\.\d+)?", raw)  # 抓第一个数字
    if not m:
        raise ValueError(f"未找到数字, 输出={raw!r}")
    val = float(m.group(0))
    return round(val, 1)


# =========================
# 剂量解析与安全裁剪
# =========================
def apply_safety_limits(dosage: float, herb_name: str) -> float:
    """先按特异范围裁剪，再做全局 3–15g 兜底，并量化到 0.5g"""
    h = herb_name.replace("生", "").replace("制", "").strip()
    if h in SAFE_DOSAGE_RANGES:
        lo, hi = SAFE_DOSAGE_RANGES[h]
        val = max(lo, min(float(dosage), hi))
        return _to_half_step(val)
    val = max(GLOBAL_MIN, min(float(dosage), GLOBAL_MAX))
    return _to_half_step(val)

def get_safe_dosage(herb_name: str) -> float:
    """默认安全剂量（优先区间中值），并量化到 0.5g"""
    h = herb_name.replace("生", "").replace("制", "").strip()
    if h in SAFE_DOSAGE_RANGES:
        lo, hi = SAFE_DOSAGE_RANGES[h]
        return _to_half_step((lo + hi) / 2)
    return _to_half_step((GLOBAL_MIN + GLOBAL_MAX) / 2)


# =========================
# 逐味生成整方剂量（核心）
# =========================
def predict_dosages(symptoms: str, herb_list: List[str]) -> Dict[str, float]:
    """
    逐味生成：每次只向 LLM 要一个数字，随后先按该味区间夹取，再做安全裁剪与去同量检查。
    """
    model, tokenizer = load_local_llm()
    print("[LLM] loaded:", model is not None)
    if model is None or tokenizer is None:
        return {h: apply_safety_limits(get_safe_dosage(h), h) for h in herb_list}

    out: Dict[str, float] = {}
    for h in herb_list:
        nm = h.replace("生", "").replace("制", "").strip()
        if nm in SAFE_DOSAGE_RANGES:
            lo, hi = SAFE_DOSAGE_RANGES[nm]
        else:
            lo, hi = GLOBAL_MIN, GLOBAL_MAX  # 若缺失，尽量完善外部表/基线
        try:
            num = _gen_number_for_herb(model, tokenizer, _single_herb_prompt(symptoms, h, lo, hi))
            # 先按该味区间夹取（避免被全局下限抬高）
            # print(_single_herb_prompt(symptoms, h, lo, hi))
            num = max(lo, min(num, hi))
        except Exception as e:
            print(f"[LLM:num] {h} 生成失败，原因: {e}，用安全中值")
            num = (lo + hi) / 2.0
        val = apply_safety_limits(num, h)  # 原有安全裁剪
        val = _to_half_step(val)  # ★ 新增：量化到 0.5g
        out[h] = val

    # —— 去同量检查：如果几乎都一样，再对“君/臣/佐/使”按比例微调 —— #
    vals = [out[h] for h in herb_list]
    if len(vals) >= 4:
        mn, mx = min(vals), max(vals)
        if (mx - mn) <= 0.2:  # 差距太小，视为“同量”
            # 简单比例：君:臣:佐:使 = 1.0 : 0.8 : 0.6 : 0.4
            ratios = [1.0, 0.8, 0.6, 0.4]
            base = max(vals[0], GLOBAL_MIN)  # 至少3g作为基准；后续再裁剪一次
            for i, name in enumerate(herb_list[:4]):
                out[name] = apply_safety_limits(round(base * ratios[i], 1), name)


    return out


# =========================
# 旧的单味剂量接口（保留以兼容可能的调用）
# =========================
@lru_cache(maxsize=100)
def predict_dosage(symptoms: str, herb_name: str) -> float:
    return apply_safety_limits(get_safe_dosage(herb_name), herb_name)

def _to_node_attrs(d):
    """
    规范化节点属性，确保包含：
    ID / name / value / output_type / node_type / dosage
    """
    return {
        "ID":        int(d.get("ID", 0)),
        "name":      str(d.get("name", "")),
        "value":     float(d.get("value", 0.0)),
        "output_type": str(d.get("output_type", "recommend")),
        "node_type":   str(d.get("node_type", "herb")),
        "dosage":    float(d.get("dosage", 0.0)),
    }

def export_node_link_json(result_dict, output_path):
    """
    result_dict: predict() 生成的 result
      {
        "input_symptoms": "症状A,症状B",
        "topk": 10,
        "herbs": [ {ID,name,value,output_type,node_type,dosage}, ... ]
      }
    output_path: 例如 ../data/output_herbs.json
    """
    # 1) 载入词表，拿到 symptom / herb 的全量序（用于 ID 对齐）
    vocab_file = join(world.FILE_PATH, "export", "vocab.json")
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sym_names = vocab["symptoms"]
    herb_names = vocab["herbs"]

    # 2) 构图
    G = nx.Graph()

    # 2.1 加 symptom 节点
    input_syms = [s.strip() for s in result_dict.get("input_symptoms", "").split(",") if s.strip()]
    for s in input_syms:
        if s in sym_names:
            sid = sym_names.index(s) + 1   # 行号/索引对齐：1-based
        else:
            # 未登录词表的症状：给一个0或跳过，这里选择跳过以避免脏节点
            continue
        G.add_node(
            s,
            **_to_node_attrs({
                "ID": sid, "name": s,
                "value": 0.0,
                "output_type": "graph",
                "node_type": "symptom",
                "dosage": 0.0
            })
        )

    # 2.2 加 herb 节点
    for h in result_dict.get("herbs", []):
        G.add_node(h["name"], **_to_node_attrs(h))

    # 2.3 加边：每个 symptom 与每个 herb 相连（无向）
    for s in input_syms:
        if s not in G:  # 过滤掉未加入的症状（如不在词表）
            continue
        for h in result_dict.get("herbs", []):
            G.add_edge(s, h["name"])

    # 3) 导出 node-link JSON
    data = nx.node_link_data(G)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def predict(symptoms_input: str, top_k: int = 10) -> Dict[str, Any]:
    """
    根据症状输入预测草药（先召回/排序，再由 LLM 逐味生成剂量）
    返回:
        {
          "input_symptoms": "...",
          "topk": top_k,
          "herbs": [{"ID":..., "name":..., "value":..., "dosage":...}, ...]
        }
    """
    result: Dict[str, Any] = {"input_symptoms": symptoms_input, "topk": top_k, "herbs": []}

    # 载入词表
    vocab_file = join(world.FILE_PATH, "export", "vocab.json")
    try:
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        sym_names = vocab["symptoms"]
        herb_names = vocab["herbs"]
    except Exception as e:
        print(f"加载词汇表失败: {e}")
        return result

    # 解析输入症状
    input_syms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    sym_indices: List[int] = []
    for s in input_syms:
        if s in sym_names:
            sym_indices.append(sym_names.index(s))
        else:
            print(f"警告: 症状 '{s}' 未在词表中找到，已忽略")
    if not sym_indices:
        print("错误: 没有有效的输入症状")
        return result

    # 加载 GCN 模型并计算相似度
    Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    try:
        Recmodel.load_state_dict(torch.load(GCN_WEIGHT_PATH, map_location=world.device))
        print(f"成功加载模型权重: {GCN_WEIGHT_PATH}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return result

    with torch.no_grad():
        Recmodel.eval()
        all_users, all_items = Recmodel.computer()

        symptom_embeddings = all_users[:len(sym_names)]
        input_emb = symptom_embeddings[sym_indices]
        combined_embedding = torch.mean(input_emb, dim=0)

        herb_embeddings = all_items
        sims = cosine_similarity(combined_embedding.unsqueeze(0), herb_embeddings)

        top_scores, top_indices = torch.topk(sims, k=top_k)
        top_indices = top_indices.cpu().numpy()
        top_scores = top_scores.cpu().numpy()
        picked_herbs = [herb_names[int(i)] for i in top_indices]

        # 一次性逐味生成剂量
        name2dose = predict_dosages(symptoms_input, picked_herbs)

    # 汇总结果
    for i in range(len(top_indices)):
        idx = int(top_indices[i])
        score = float(top_scores[i])
        herb_name = herb_names[idx]
        dosage = name2dose.get(herb_name, get_safe_dosage(herb_name))

        result["herbs"].append({
            "ID": int(idx + 1),
            "name": herb_name,
            "value": score,
            "output_type": "recommend",
            "node_type": "herb",
            "dosage": dosage
        })

    out_path = join("..", "data", "output_herbs.json")
    export_node_link_json(result, out_path)

    return result

def symptom_input(xlsx_path: str = r"D:\Program Files\code\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code\input_symptom.xlsx",
                  top_k: int = 10,
                  row: int = 0):
    """
    读取 Excel 第一列第 row 行：
      - 如果是“空格/逗号分隔的编号（1-based）”，按编号解析；
      - 如果不是编号，就按“症状名称（支持中文/英文逗号、空格、顿号分隔）”解析；
    然后转成 predict() 需要的“逗号分隔的症状名”，调用现有流程。
    最终 node-link JSON 的导出仍由 predict() 内部完成（../data/output_herbs.json）。
    """
    # 1) 载入词表
    vocab_file = join(world.FILE_PATH, "export", "vocab.json")
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sym_names = vocab["symptoms"]
    N = len(sym_names)

    # 2) 读取 Excel 目标行
    try:
        df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
    except Exception:
        df = pd.read_excel(xlsx_path, header=None)
    raw = df.iloc[row, 0] if row < len(df) else ""
    s = "" if pd.isna(raw) else str(raw)

    # 3) 统一分隔符（全角空格/中文逗号/英文逗号/顿号/制表符 -> 空格）
    for bad, good in [("\u3000", " "), ("，", " "), (",", " "), ("、", " "), ("\t", " ")]:
        s = s.replace(bad, good)
    parts = [p for p in s.strip().split() if p]

    # 4) 先尝试按“编号”解析（1-based → 0-based）
    idx0 = []
    if parts:
        nums = []
        for p in parts:
            if p.isdigit():
                nums.append(int(p))
            else:
                # 允许像 "12," 这种混入符号的情况
                digits = "".join(ch for ch in p if ch.isdigit())
                if digits:
                    try:
                        nums.append(int(digits))
                    except Exception:
                        pass
        idx0 = [i - 1 for i in nums if 1 <= i <= N]

    # 5) 若没解析出编号，则按“名称”解析
    input_syms = []
    if idx0:
        input_syms = [sym_names[i] for i in idx0]
        print(f"[symptom_input] 行 {row} 解析为编号: {[i+1 for i in idx0]}")
    else:
        # 名称模式：把 parts 当成症状名（已去除各种分隔符）
        found, unknown = [], []
        for token in parts:
            if token in sym_names:
                found.append(token)
            else:
                unknown.append(token)
        input_syms = found
        if unknown:
            print(f"[symptom_input] 注意: 这些名称不在词表中，已忽略: {unknown}")
        if not found:
            print(f"[symptom_input] 错误: 第 {row} 行既不是有效编号，也没有匹配到任何症状名称。原始: {raw!r}")
            return {"input_symptoms": "", "topk": top_k, "herbs": []}, join("..", "data", "output_herbs.json")

        print(f"[symptom_input] 行 {row} 解析为名称: {found}")

    symptoms_input = ",".join(input_syms)
    result = predict(symptoms_input, top_k=top_k)
    return result, join("..", "data", "output_herbs.json")




if __name__ == "__main__":
    # 你的工程里这些全局在 world 内部使用
    world.dataset = "prescription"
    world.model_name = "lightgcn_cooccur"

    res, out_json = symptom_input("D:\Program Files\code\LightGCN-PyTorch-master\LightGCN-PyTorch-master\code\input_symptom.xlsx", top_k=10, row=0)

    print("\n推荐结果:")
    print(f"输入症状: {res['input_symptoms']}")
    for herb in res["herbs"]:
        print(f"草药ID: {herb['ID']}, 名称: {herb['name']}, 分数: {herb['value']:.4f}, 剂量: {herb['dosage']}克")

    output_file = join(world.FILE_PATH, "export", "output_prescription.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")

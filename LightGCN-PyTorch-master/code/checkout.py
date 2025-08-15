import os
import torch
from transformers import AutoModel, AutoTokenizer

# 可选：关闭 Windows 符号链接警告
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

model_name_or_path = "scutcyr/BianQue-2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) 模型 —— 注意用 AutoModel 而不是 AutoModelForCausalLM
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
if device.type == "cuda":
    model = model.half().to(device)
else:
    model = model.to(device)
model.eval()

def call_bianque_json(symptoms: str, herbs: list[str]) -> str:
    # 紧凑、刚性的提示（只要 JSON）
    herbs_block = "\n".join([f"- {h}" for h in herbs])
    prompt = (
        "任务: 依据所有症状与下列草药的整体配伍(君臣佐使/协同拮抗/四气五味/升降浮沉/毒性禁忌),"
        "一次性分配每味药的单日煎服剂量(克, 1位小数)。严禁全部相同剂量,只能使用清单内药名。\n"
        f"症状:\n{symptoms}\n"
        f"草药(顺序固定):\n{herbs_block}\n"
        "只输出一个JSON对象，键为\"dosages\"，值为与清单等长且同序的数组，元素形如"
        "{\"name\":\"木香\",\"dosage\":6.0}。只输出JSON，无任何解释。\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    # ChatGLM 在 transformers 里 attention_mask 用 bool 更稳
    inputs["attention_mask"] = inputs["attention_mask"].bool()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        # ChatGLM 的 remote code 已实现 generate；这里走贪心，输出更规整
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# 示例
if __name__ == "__main__":
    symptoms = "小儿惊疳"
    herbs = ["重楼","使君子","铅霜","走石","蛇黄","天竺黄","麝香","胡黄连","熊胆","朱砂"]
    json_text = call_bianque_json(symptoms, herbs)
    print(json_text)

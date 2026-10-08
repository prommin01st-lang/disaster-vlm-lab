"""LoRA SFT บน Qwen3.5-2B (bf16, Colab L4) — loss เฉพาะ token คำตอบ
env: MAX_STEPS (smoke; ถ้า >0 ไม่อัป HF), EPOCHS (default 2), LR (default 1e-4)"""
import json, os, tarfile
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from dvl.prompt import build_messages

MODEL = "Qwen/Qwen3.5-2B"
SEED = 20261007
MAX_STEPS = int(os.environ.get("MAX_STEPS", -1))
api = HfApi(); user = api.whoami()["name"]
tarfile.open(hf_hub_download(f"{user}/dvl-data", "dataset_v1.tar", repo_type="dataset")).extractall("data", filter="data")
D = Path("data/dataset_v1")


def load(name):
    return [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8")]


train_rows, val_rows = load("train"), load("val")
processor = AutoProcessor.from_pretrained(MODEL)
processor.tokenizer.padding_side = "right"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

# ชื่อโมดูล linear ของ language model — Qwen3.5 มี linear-attention (in_proj_qkv/in_proj_z/out_proj)
# ปนกับ full-attention (q/k/v/o_proj) จึงต้องใส่ทั้งสองชุด (in_proj_a/b เล็กมาก ข้าม)
names = sorted({n.split(".")[-1] for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and "visual" not in n})
print("linear module names (non-vision):", names)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
                   r"|in_proj_qkv|in_proj_z|out_proj)$"))
model.print_trainable_parameters()

TEMPLATE_KW = dict(tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)


def collate(batch):
    # processor ของ Qwen3.5 ไม่รับ images= ใน apply_chat_template → ใส่ภาพลงใน message (เหมือนตอน predict)
    images = [Image.open(D / r["image"]).convert("RGB") for r in batch]
    full = [build_messages(im) + [{"role": "assistant", "content": [{"type": "text", "text": r["target"]}]}]
            for im, r in zip(images, batch)]
    enc = processor.apply_chat_template(full, padding=True, **TEMPLATE_KW)
    labels = enc["input_ids"].clone()
    for i, im in enumerate(images):  # prompt = ทุกอย่างก่อนคำตอบ (รวม <think></think> ว่าง) → ไม่คิด loss
        plen = processor.apply_chat_template([build_messages(im)], add_generation_prompt=True,
                                             **TEMPLATE_KW)["input_ids"].shape[1]
        labels[i, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


args = SFTConfig(
    output_dir="out/ckpt", per_device_train_batch_size=4, per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, num_train_epochs=float(os.environ.get("EPOCHS", 2)),
    max_steps=MAX_STEPS, learning_rate=float(os.environ.get("LR", 1e-4)),
    lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, fp16=False,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10 if MAX_STEPS < 0 else 2,
    eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", report_to="none",
    remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True}, seed=SEED)
# train.jsonl เรียงตามคลาส — สลับเองด้วย seed (Trainer ก็สุ่ม sampler อีกชั้น)
trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows).shuffle(seed=SEED),
                     eval_dataset=Dataset.from_list(val_rows),
                     data_collator=collate, processing_class=processor,
                     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
trainer.train()
if MAX_STEPS > 0:
    print("smoke final eval:", trainer.evaluate(eval_dataset=Dataset.from_list(val_rows[:64])))
print("best checkpoint:", trainer.state.best_model_checkpoint, "best eval_loss:", trainer.state.best_metric)
json.dump(trainer.state.log_history, open("out/log_history.json", "w"), indent=1)
model.save_pretrained("out/adapter")
if MAX_STEPS < 0:
    repo = f"{user}/dvl-qwen35-2b-lora"
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path="out/adapter", repo_id=repo)
    print("uploaded →", repo)

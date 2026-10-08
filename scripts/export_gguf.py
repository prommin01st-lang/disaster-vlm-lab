"""รวม LoRA เข้า Qwen3.5-2B → GGUF (Q8_0 / Q4_K_M + mmproj F16) → อัป HF private (รันบน Colab CPU)
env: ADAPTER (default Petanque/dvl-qwen35-2b-lora), GGUF_REPO (default <user>/dvl-qwen3.5-2b-gguf),
     LLAMA_DIR (source clone, มี convert_hf_to_gguf.py), LLAMA_QUANTIZE (path ไบนารี llama-quantize)
`python scripts/export_gguf.py card` = เขียน/อัป README การ์ดอย่างเดียว (ใช้ตอนเติมผล eval จากเครื่อง local)"""
import json, os, subprocess, sys
from pathlib import Path

from huggingface_hub import HfApi

from dvl.catalog import ALLOWED_TYPES
from dvl.prompt import SYSTEM_PROMPT, USER_INSTRUCTION

BASE = "Qwen/Qwen3.5-2B"
ADAPTER = os.environ.get("ADAPTER", "Petanque/dvl-qwen35-2b-lora")
NAME = "dvl-qwen3.5-2b"
FILES = {"q8": f"{NAME}-Q8_0.gguf", "q4": f"{NAME}-Q4_K_M.gguf", "mmproj": f"mmproj-{NAME}-F16.gguf"}
LORA_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
               "in_proj_qkv", "in_proj_z", "out_proj")


def merge(dst: Path) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, ADAPTER)
    # นับ module ที่มี LoRA ต่อชนิด — linear-attention (in_proj_qkv/in_proj_z/out_proj) ต้องมีครบ 18 ชั้น
    counts = {k: 0 for k in LORA_LINEAR}
    probe = None
    for n, m in model.named_modules():
        if hasattr(m, "lora_A") and n.split(".")[-1] in counts:
            counts[n.split(".")[-1]] += 1
            if probe is None and n.endswith("in_proj_qkv"):
                probe = (n, m.base_layer.weight.detach().clone())
    print("lora modules per type:", counts, flush=True)
    assert counts["in_proj_qkv"] == counts["in_proj_z"] == counts["out_proj"] == 18, counts
    assert counts["q_proj"] >= 6, counts  # 6 full-attention layers (+ MTP ถ้า transformers โหลดมา)
    model = model.merge_and_unload()
    merged_w = dict(model.named_parameters())[probe[0].replace("base_model.model.", "") + ".weight"]
    delta = (merged_w.float() - probe[1].float()).abs().max().item()
    print(f"merge check {probe[0]}: max|Δw| = {delta:.3e}", flush=True)
    assert delta > 0, "LoRA merge did not change linear-attention weights"
    model.save_pretrained(dst, safe_serialization=True)
    AutoProcessor.from_pretrained(BASE).save_pretrained(dst)  # tokenizer + chat_template + image processor
    cfg = json.loads((dst / "config.json").read_text())
    return {"lora_modules": counts, "probe_max_delta": delta,
            "tie_word_embeddings": cfg.get("text_config", cfg).get("tie_word_embeddings")}


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def tensor_summary(path: Path) -> dict:
    from gguf import GGUFReader
    r = GGUFReader(path)
    names = [t.name for t in r.tensors]
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    return {"arch": arch, "n_tensors": len(names), "has_output_weight": "output.weight" in names,
            "has_token_embd": "token_embd.weight" in names}


def card(sizes: dict[str, int], llama_commit: str) -> str:
    types = ", ".join(f"`{k}`" for k in ALLOWED_TYPES)
    rows = "\n".join(f"| `{FILES[k]}` | {sizes[k] / 2**30:.2f} GiB |" for k in ("q8", "q4", "mmproj") if k in sizes)
    results = []
    for q in ("q8", "q4"):
        p = Path(f"reports/gguf_{q}-gold.json")
        if p.exists():
            m = json.loads(p.read_text())
            st = Path(f"runs/gguf/stats-gguf_{q}.json")
            st = json.loads(st.read_text()) if st.exists() else {}
            results.append(f"| {FILES[q]} | {m['type_macro_f1']:.3f} | {m['false_alarm_rate']:.3f} | "
                           f"{m['miss_rate']:.3f} | {m['json_valid_rate']:.3f} | {st.get('s_per_img', '?')} | "
                           f"{st.get('vram_peak_mib_total', '?')} |")
    res = ("Gold set: 247 human-reviewed, class-balanced test images; llama.cpp b10909 Vulkan on an RTX 2060 6GB, "
           "temperature 0. VRAM is the whole card as reported by nvidia-smi (desktop uses about 350 MiB). "
           "Un-tuned Qwen3.5-2B (bf16, transformers) scores macro-F1 0.525 / false_alarm 0.400 on the same set. "
           "Full reports: `reports/gguf_q8-gold.json`, `reports/gguf_q4-gold.json`.\n\n"
           "| file | macro-F1 | false_alarm | miss | json_valid | s/img | VRAM peak (MiB) |\n"
           "|---|---|---|---|---|---|---|\n" + "\n".join(results)
           if results else "_Pending._ Local llama.cpp evaluation on the 247-row human-reviewed gold set is "
                           "written to `reports/gguf_q8-gold.json` / `reports/gguf_q4-gold.json` in the project repo.")
    return f"""---
base_model: {BASE}
tags: [gguf, llama.cpp, vision, qwen3.5, lora-merged, disaster]
license: other
license_name: cc-by-nc-sa-4.0-training-data
---

# dvl-qwen3.5-2b-gguf

GGUF build of [`{BASE}`](https://huggingface.co/{BASE}) with the LoRA adapter
[`{ADAPTER}`](https://huggingface.co/{ADAPTER}) (best checkpoint, step 700) merged in bf16.
The model looks at an emergency-report photo and answers one line of JSON. Built for
llama.cpp on small GPUs (tested on an RTX 2060 6GB, Vulkan).

Converted with llama.cpp `{llama_commit}` (`convert_hf_to_gguf.py --no-mtp`, then `llama-quantize`).
The MTP head is not exported (only used for speculative decoding). Qwen3.5 ties `lm_head` to the
token embeddings; the GGUF has no separate `output.weight` and llama.cpp reuses `token_embd.weight`.

## Files

| file | size |
|---|---|
{rows}

Q8_0 is the recommended default; Q4_K_M is smaller. You always need the `mmproj` file for images.

## Output schema

```json
{{"category": "...", "incident_type": "...", "severity": "none|mild|severe"}}
```

`incident_type` is one of: {types}. `category` is derived from `incident_type`
(`null` for `no_incident`). The model does not emit a confidence value. The project computes
confidence as the product of the token probabilities of the `incident_type` value
(request `logprobs` from llama-server; see `scripts/predict_gguf.py`).

## Prompt (must match exactly)

Thinking must be off (`enable_thinking: false`). System prompt:

```text
{SYSTEM_PROMPT}
```

User turn: the image first, then the text `{USER_INSTRUCTION}`.

## Usage

```bash
llama-server -m {FILES['q8']} --mmproj {FILES['mmproj']} -ngl 99 -c 4096 -np 1 \\
  --jinja --reasoning off --port 8080
```

Then POST to `http://localhost:8080/v1/chat/completions` with the system prompt above, a user message
containing an `image_url` (base64 data URL) followed by the text, `temperature: 0`, `max_tokens: 64`,
and `"chat_template_kwargs": {{"enable_thinking": false}}`. Add `"logprobs": true` to get token
probabilities for the confidence value.

Quick smoke test from the CLI. Use llama-server for real runs: `llama-mtmd-cli --jinja` fails with
`-sys` on this template ("No user query found"), and without `--jinja` its built-in template lets the
model print an empty `<think></think>` block before the JSON, so the prompt does not match training exactly.

```bash
llama-mtmd-cli -m {FILES['q8']} --mmproj {FILES['mmproj']} -ngl 99 -c 4096 --temp 0 -n 64 \\
  -sys "$(cat system_prompt.txt)" --image photo.jpg -p "{USER_INSTRUCTION}"
```

## Results

{res}

## Training data and licenses

The adapter was trained on images from: QCRI/CrisisMMD (CC BY-NC-SA 4.0), anwan/DisasterVQA
(CC BY-SA 4.0), AbdullahImran/balanced_wildfire_dataset (other),
fireviewer/fire_and_smoke_detection_very_hard_negative (other), hiennguyen9874/traffic-accident-detection
(no license stated), Arpitraj01/Pothole_classification (MIT), nlphuji/flickr_1k_test_image_text_retrieval
(no license stated). Labels come from source-dataset mappings plus a Qwen3.5-9B teacher.

**Non-commercial:** because CrisisMMD is CC BY-NC-SA 4.0, treat these weights as non-commercial,
share-alike. The base model Qwen3.5-2B is Apache-2.0.

This is a learning project. Do not use it as the only signal for real emergency dispatch.
"""


def upload_card(api: HfApi, repo: str, sizes: dict, commit: str) -> None:
    api.upload_file(path_or_fileobj=card(sizes, commit).encode(), path_in_repo="README.md", repo_id=repo)


def main() -> None:
    api = HfApi()
    repo = os.environ.get("GGUF_REPO") or f"{api.whoami()['name']}/{NAME}-gguf"
    if sys.argv[1:] == ["card"]:  # local: เติมผล eval ลงการ์ด (ขนาดไฟล์อ่านจาก repo)
        info = api.model_info(repo, files_metadata=True)
        by_name = {s.rfilename: s.size for s in info.siblings}
        sizes = {k: by_name[f] for k, f in FILES.items() if f in by_name}
        commit = os.environ.get("LLAMA_COMMIT", "b10909")
        upload_card(api, repo, sizes, commit)
        print("card updated →", repo)
        return

    llama = Path(os.environ["LLAMA_DIR"])
    quant = os.environ["LLAMA_QUANTIZE"]
    commit = subprocess.run(["git", "-C", llama, "describe", "--tags", "--always"], capture_output=True,
                            text=True).stdout.strip() + " (" + subprocess.run(
        ["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:9] + ")"
    work = Path("/content/gguf"); work.mkdir(exist_ok=True)
    merged = work / "merged"
    info = {"llama_cpp": commit, "adapter": ADAPTER, "repo": repo, **merge(merged)}
    bf16 = work / f"{NAME}-BF16.gguf"
    conv = [sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged)]
    run(conv + ["--outtype", "bf16", "--no-mtp", "--outfile", str(bf16)])
    run(conv + ["--mmproj", "--outtype", "f16", "--outfile", str(work / FILES["mmproj"])])
    run([quant, str(bf16), str(work / FILES["q8"]), "Q8_0"])
    run([quant, str(bf16), str(work / FILES["q4"]), "Q4_K_M"])
    sizes = {k: (work / f).stat().st_size for k, f in FILES.items()}
    info["sizes"] = sizes | {"bf16": bf16.stat().st_size}
    info["gguf"] = {k: tensor_summary(work / f) for k, f in FILES.items()}
    print(json.dumps(info, indent=2), flush=True)

    api.create_repo(repo, private=True, exist_ok=True)
    for f in FILES.values():
        api.upload_file(path_or_fileobj=str(work / f), path_in_repo=f, repo_id=repo)
        print("uploaded", f, flush=True)
    upload_card(api, repo, sizes, commit)
    Path("out/export_gguf.json").write_text(json.dumps(info, indent=2))
    print("done →", repo)


if __name__ == "__main__":
    main()

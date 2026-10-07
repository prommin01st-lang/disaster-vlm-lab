"""โหลด VLM + generate แบบ greedy พร้อม confidence — ใช้ร่วมกันทั้ง teacher และ predict"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from dvl.confidence import span_confidence


def load_model(model_id: str, adapter: str | None = None):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_json(model, processor, messages: list[dict]) -> tuple[str, float | None]:
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    probs = [torch.softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new_ids)]
    tok = processor.tokenizer
    pieces = [tok.decode([t], skip_special_tokens=False) for t in new_ids]
    keep = [i for i, t in enumerate(new_ids) if t not in tok.all_special_ids]
    text = "".join(pieces[i] for i in keep)
    return text, span_confidence([pieces[i] for i in keep], [probs[i] for i in keep])

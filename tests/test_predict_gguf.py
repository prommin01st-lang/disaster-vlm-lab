import base64, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from predict_gguf import parse_response, request_body  # noqa: E402

from dvl.prompt import SYSTEM_PROMPT, USER_INSTRUCTION, build_messages  # noqa: E402


def test_request_matches_build_messages(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8fake")
    body = request_body(build_messages(img))
    sys_msg, user = body["messages"]
    assert sys_msg == {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    assert user["role"] == "user"
    image, text = user["content"]  # ภาพก่อนข้อความ เหมือน build_messages
    assert image["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8fake").decode()
    assert text == {"type": "text", "text": USER_INSTRUCTION}
    assert body["temperature"] == 0 and body["max_tokens"] == 64 and body["logprobs"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def _resp(tokens, probs, reasoning=None):
    return {"choices": [{"message": {"content": "".join(tokens), "reasoning_content": reasoning},
                         "logprobs": {"content": [{"token": t, "logprob": math.log(p)}
                                                  for t, p in zip(tokens, probs)]}}]}


def test_confidence_from_logprobs():
    toks = ['{"', 'category', '":"', 'fire', '","', 'incident_type', '":"', 'forest', '_fire', '","',
            'severity', '":"', 'severe', '"}']
    probs = [1.0] * len(toks)
    probs[7], probs[8] = 0.8, 0.5
    text, conf, think = parse_response(_resp(toks, probs))
    assert json.loads(text)["incident_type"] == "forest_fire"
    assert math.isclose(conf, 0.4)
    assert think is False


def test_flags_thinking_and_missing_logprobs():
    r = _resp(['<think>', 'x', '</think>', '{}'], [1.0] * 4)
    assert parse_response(r)[2] is True
    r = {"choices": [{"message": {"content": "{}", "reasoning_content": "hmm"}}]}
    _, conf, think = parse_response(r)
    assert conf is None and think is True

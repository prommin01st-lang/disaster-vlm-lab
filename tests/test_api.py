"""api/app.py — llama-server จำลองด้วย httpx.MockTransport (ไม่ใช้ GPU)"""
import asyncio, base64, io, json, math, warnings

import httpx
import pytest
from PIL import Image

with warnings.catch_warnings():  # starlette 1.7: "Using httpx with starlette.testclient is deprecated"
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from api.app import Settings, create_app
from dvl.prompt import SYSTEM_PROMPT

KEYS = {"category", "incident_type", "severity", "confidence", "p_incident", "needs_review", "review_reason",
        "incident_type_name_th", "valid", "model", "threshold", "latency_ms"}


def _png(size=(64, 48), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (10, 120, 200, 128)[: len(mode)]).save(buf, format="PNG")
    return buf.getvalue()


def _chat(text_tokens: list[str], cat_alts: list[tuple[str, float]]) -> dict:
    """คำตอบแบบ llama-server: logprobs ต่อ token; top_logprobs ของ token แรกของค่า category = cat_alts (prob)"""
    lp, pos = [], 0
    text = "".join(text_tokens)
    value_start = text.index('"category":') + len('"category":')
    value_start += text[value_start] == '"'
    for t in text_tokens:
        top = [{"token": a, "logprob": math.log(p)} for a, p in cat_alts] if pos == value_start else []
        lp.append({"token": t, "logprob": math.log(0.99), "top_logprobs": top})
        pos += len(t)
    return {"choices": [{"message": {"content": text}, "logprobs": {"content": lp}}]}


NO_INCIDENT = ['{"', 'category', '":', 'null', ',"', 'incident_type', '":"', 'no', '_incident', '","',
               'severity', '":"', 'none', '"}']
FIRE = ['{"', 'category', '":"', 'fire', '_hazard', '","', 'incident_type', '":"', 'forest', '_fire', '","',
        'severity', '":"', 'severe', '"}']
UNSURE = ['{"', 'category', '":"', 'other', '","', 'incident_type', '":"', 'unsure', '","',
          'severity', '":"', 'mild', '"}']


class FakeLlama:
    def __init__(self, chat=None, health=200, exc=None):
        self.chat, self.health, self.exc, self.bodies = chat, health, exc, []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.exc:
            raise self.exc
        if req.url.path == "/health":
            return httpx.Response(self.health, json={"status": "ok" if self.health == 200 else "loading model"})
        self.bodies.append(json.loads(req.content))
        return httpx.Response(200, json=self.chat)


def client(fake: FakeLlama, **kw) -> TestClient:
    s = Settings(llama_url="http://llama.test", abstain_t=3.2e-5, model_name="dvl-test", api_key=kw.pop("api_key", None),
                 timeout=5.0)
    return TestClient(create_app(s, transport=httpx.MockTransport(fake)))


def post_img(c, data=None, ctype="image/png", headers=None):
    return c.post("/v1/classify", files={"image": ("x.png", data if data is not None else _png(), ctype)},
                  headers=headers or {})


def test_happy_path_incident_and_preprocessing():
    fake = FakeLlama(_chat(FIRE, [("fire", 0.97), ("dis", 0.03)]))
    r = post_img(client(fake), _png((1024, 600), "RGBA"))
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == KEYS
    assert (j["category"], j["incident_type"], j["severity"]) == ("fire_hazard", "forest_fire", "severe")
    assert j["needs_review"] is False and j["review_reason"] is None and j["valid"] is True
    assert j["incident_type_name_th"] == "ไฟป่า"
    assert math.isclose(j["confidence"], 0.98, abs_tol=1e-3)  # 0.99 * 0.99 (forest, _fire)
    assert j["p_incident"] == 1.0 and j["model"] == "dvl-test" and j["threshold"] == 3.2e-5
    assert isinstance(j["latency_ms"], int)
    # ข้อความตรง dvl.prompt + ภาพถูกย่อเป็น RGB JPEG ≤ 512
    sys_msg, user = fake.bodies[0]["messages"]
    assert sys_msg["content"][0]["text"] == SYSTEM_PROMPT
    url = user["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.format == "JPEG" and img.mode == "RGB" and img.size == (512, 300)
    assert fake.bodies[0]["temperature"] == 0 and fake.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_confident_no_incident_passes():
    fake = FakeLlama(_chat(NO_INCIDENT, [("null", 1 - 1e-6), (' "', 1e-6)]))
    j = post_img(client(fake)).json()
    assert (j["category"], j["incident_type"], j["severity"]) == (None, "no_incident", "none")
    assert j["needs_review"] is False and j["review_reason"] is None
    assert j["p_incident"] < 3.2e-5


def test_gate_fires_on_shaky_no_incident():
    fake = FakeLlama(_chat(NO_INCIDENT, [("null", 0.999), (' "', 0.001)]))
    j = post_img(client(fake)).json()
    assert (j["category"], j["incident_type"], j["severity"]) == ("other", "unsure", "mild")
    assert j["needs_review"] is True and j["review_reason"] == "uncertain_no_incident"
    assert math.isclose(j["p_incident"], 0.001, rel_tol=1e-3)
    assert j["incident_type_name_th"] == "ไม่แน่ใจประเภทเหตุ"


def test_model_unsure():
    j = post_img(client(FakeLlama(_chat(UNSURE, [("other", 0.6), ("dis", 0.4)])))).json()
    assert j["incident_type"] == "unsure" and j["needs_review"] is True and j["review_reason"] == "model_unsure"
    assert j["valid"] is True


def test_invalid_model_output():
    chat = {"choices": [{"message": {"content": "ขอโทษ ดูไม่ออก"}, "logprobs": {"content": []}}]}
    j = post_img(client(FakeLlama(chat))).json()
    assert j["valid"] is False and j["incident_type"] == "unsure"
    assert j["needs_review"] is True and j["review_reason"] == "invalid_output"
    assert j["confidence"] == 0.0


@pytest.mark.parametrize("ctype,data,status,code", [
    ("image/gif", None, 415, "unsupported_media_type"),
    ("image/png", b"not an image at all", 400, "undecodable_image"),
    ("image/png", b"", 400, "undecodable_image"),
])
def test_bad_uploads(ctype, data, status, code):
    fake = FakeLlama(_chat(FIRE, [("fire", 1.0)]))
    r = post_img(client(fake), data, ctype)
    assert r.status_code == status and r.json()["error"] == code
    assert fake.bodies == []


def test_gif_bytes_with_png_content_type_rejected():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="GIF")
    r = post_img(client(FakeLlama()), buf.getvalue(), "image/png")
    assert r.status_code == 415 and r.json()["error"] == "unsupported_media_type"


def test_missing_image_field():
    r = client(FakeLlama()).post("/v1/classify", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 400 and r.json()["error"] == "missing_image"


def test_oversized():
    r = post_img(client(FakeLlama()), b"\xff" * (10 * 1024 * 1024 + 1), "image/jpeg")
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_oversized_rejected_by_content_length_before_parsing():
    c = client(FakeLlama())
    r = c.post("/v1/classify", content=b"x", headers={"Content-Length": str(50 * 1024 * 1024),
                                                      "Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_api_key_required_when_set():
    c = client(FakeLlama(_chat(FIRE, [("fire", 1.0)])), api_key="s3cret")
    assert post_img(c).status_code == 401
    assert post_img(c, headers={"X-API-Key": "nope"}).json()["error"] == "unauthorized"
    assert c.get("/v1/labels").status_code == 401
    assert post_img(c, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200  # /health ไม่ต้องใช้ key


def test_api_key_off_by_default():
    assert client(FakeLlama()).get("/v1/labels").status_code == 200


def test_labels():
    labels = client(FakeLlama()).get("/v1/labels").json()["labels"]
    by = {l["incident_type"]: l for l in labels}
    assert len(labels) == 14
    assert by["flood"] == {"incident_type": "flood", "category": "disaster", "name_th": "น้ำท่วม / น้ำป่าไหลหลาก"}
    assert by["no_incident"]["category"] is None and by["unsure"]["category"] == "other"


def test_health_ok_and_loading():
    assert client(FakeLlama(health=200)).get("/health").status_code == 200
    r = client(FakeLlama(health=503)).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"


def test_llama_down():
    c = client(FakeLlama(exc=httpx.ConnectError("refused")))
    assert c.get("/health").status_code == 503
    r = post_img(c)
    assert r.status_code == 502 and r.json()["error"] == "llama_unreachable"


def test_llama_timeout():
    r = post_img(client(FakeLlama(exc=httpx.ReadTimeout("slow"))))
    assert r.status_code == 504 and r.json()["error"] == "llama_timeout"


def test_llama_error_status():
    def handler(req):
        return httpx.Response(500, json={"error": "boom"})
    c = TestClient(create_app(Settings(llama_url="http://l"), transport=httpx.MockTransport(handler)))
    r = post_img(c)
    assert r.status_code == 502 and r.json()["error"] == "llama_error"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DVL_LLAMA_URL", "http://x:1/")
    monkeypatch.setenv("DVL_ABSTAIN_T", "1e-4")
    monkeypatch.setenv("DVL_API_KEY", "k")
    monkeypatch.setenv("DVL_TIMEOUT", "9")
    s = Settings.from_env()
    assert (s.llama_url, s.abstain_t, s.api_key, s.timeout) == ("http://x:1", 1e-4, "k", 9.0)
    monkeypatch.delenv("DVL_API_KEY")
    monkeypatch.delenv("DVL_ABSTAIN_T")
    s = Settings.from_env()
    assert s.api_key is None and s.abstain_t == 3.2e-5


def test_llama_calls_are_serialized():
    state = {"active": 0, "max": 0}

    async def handler(req):
        state["active"] += 1
        state["max"] = max(state["max"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return httpx.Response(200, json=_chat(FIRE, [("fire", 1.0)]))

    app = create_app(Settings(llama_url="http://l"), transport=httpx.MockTransport(handler))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as c:
            rs = await asyncio.gather(*[c.post("/v1/classify", files={"image": ("x.png", _png(), "image/png")})
                                        for _ in range(4)])
        return [r.status_code for r in rs]

    assert asyncio.run(run()) == [200] * 4
    assert state["max"] == 1

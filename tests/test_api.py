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

KEYS = {"category", "incident_type", "model_incident_type", "severity", "confidence", "p_incident", "needs_review", "review_reason",
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
    assert j["incident_type_name_th"] == "ไฟป่า" and j["model_incident_type"] == "forest_fire"
    assert math.isclose(j["confidence"], 0.98, abs_tol=1e-3)  # 0.99 * 0.99 (forest, _fire)
    assert j["p_incident"] == 1.0 and j["model"] == "dvl-test" and j["threshold"] == 3.2e-5
    assert isinstance(j["latency_ms"], int)
    # ข้อความตรง dvl.prompt + ภาพถูกย่อเป็น RGB JPEG ≤ 512
    sys_msg, user = fake.bodies[0]["messages"]
    assert sys_msg["content"][0]["text"] == SYSTEM_PROMPT
    url = user["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.format == "PNG" and img.mode == "RGB" and img.size == (512, 300)
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
    assert j["confidence"] == 0.0 and j["model_incident_type"] is None


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


def _png_header_only(w: int, h: int) -> bytes:
    """PNG ที่ประกาศขนาด w×h (1-bit) แต่ IDAT เล็กมาก — ไฟล์ไม่กี่ร้อย byte"""
    import struct, zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b""))


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")  # app ปิด warning นี้ แต่ pytest reset filter
@pytest.mark.parametrize("w,h", [(20000, 20000), (8000, 6000), (100000, 100000)])
def test_too_many_pixels_rejected_before_decode(w, h):
    fake = FakeLlama(_chat(FIRE, [("fire", 1.0)]))
    data = _png_header_only(w, h)
    assert len(data) < 1000
    r = post_img(client(fake), data, "image/png")
    assert r.status_code == 413 and r.json()["error"] == "too_many_pixels"
    assert fake.bodies == []


def test_large_jpeg_is_draft_decoded_and_resized():
    buf = io.BytesIO()
    Image.new("RGB", (4000, 3000), (30, 60, 90)).save(buf, format="JPEG")
    fake = FakeLlama(_chat(FIRE, [("fire", 1.0)]))
    assert post_img(client(fake), buf.getvalue(), "image/jpeg").status_code == 200
    url = fake.bodies[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (512, 384)


def _chunked_multipart(total: int):
    b = b"BOUNDARY"
    yield b"--" + b + b'\r\nContent-Disposition: form-data; name="image"; filename="x.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
    sent = 0
    while sent < total:
        yield b"\xff" * 1024 * 1024
        sent += 1024 * 1024
    yield b"\r\n--" + b + b"--\r\n"


def test_chunked_body_over_limit_rejected():
    fake = FakeLlama(_chat(FIRE, [("fire", 1.0)]))
    r = client(fake).post("/v1/classify", content=_chunked_multipart(40 * 1024 * 1024),
                          headers={"Content-Type": "multipart/form-data; boundary=BOUNDARY"})
    assert "content-length" not in {k.lower() for k in r.request.headers}
    assert r.status_code == 413 and r.json()["error"] == "too_large"
    assert fake.bodies == []


def test_chunked_body_under_limit_ok():
    fake = FakeLlama(_chat(FIRE, [("fire", 1.0)]))
    png = _png()

    def gen():
        yield b'--B\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'
        yield png
        yield b"\r\n--B--\r\n"
    r = client(fake).post("/v1/classify", content=gen(), headers={"Content-Type": "multipart/form-data; boundary=B"})
    assert r.status_code == 200, r.text


def test_streamed_body_limit_stops_reading_early():
    """ASGI ตรง ๆ: body ไม่มี Content-Length ถูกนับระหว่างอ่าน → 413 และหยุดดึง body ก่อนอ่านหมด"""
    app = create_app(Settings(llama_url="http://l"), transport=httpx.MockTransport(FakeLlama()))
    chunk, pulled, sent = b"\xff" * (1024 * 1024), [0], []
    head = b'--B\r\nContent-Disposition: form-data; name="image"; filename="x.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'

    async def receive():
        pulled[0] += 1
        body = head if pulled[0] == 1 else chunk
        return {"type": "http.request", "body": body, "more_body": pulled[0] < 200}  # ~200 MB ถ้าอ่านหมด

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/classify", "raw_path": b"/v1/classify", "query_string": b"", "root_path": "",
             "headers": [(b"content-type", b"multipart/form-data; boundary=B"), (b"host", b"api")],
             "client": ("127.0.0.1", 1), "server": ("api", 80)}
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert start["status"] == 413 and json.loads(body)["error"] == "too_large"
    assert pulled[0] <= 13  # 10 MB + 64 KB แล้วหยุด


def test_png_sent_lossless_and_model_answer_exposed():
    src = Image.new("RGB", (300, 200))
    src.putdata([(x % 256, y % 256, (x * y) % 256) for y in range(200) for x in range(300)])
    buf = io.BytesIO()
    src.save(buf, format="JPEG", quality=90)
    decoded = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    fake = FakeLlama(_chat(NO_INCIDENT, [("null", 0.999), (' "', 0.001)]))
    j = post_img(client(fake), buf.getvalue(), "image/jpeg").json()
    url = fake.bodies[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    sent_img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent_img.tobytes() == decoded.tobytes()  # pixel เดียวกับที่ถอดจาก JPEG ต้นฉบับ
    assert j["incident_type"] == "unsure" and j["model_incident_type"] == "no_incident"


def test_decodes_are_bounded():
    import api.app as m
    assert m.DECODE_CONCURRENCY == 2


def test_web_ui_served_without_key():
    c = client(FakeLlama(), api_key="k")
    r = c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "/v1/classify" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"

"""HTTP API: ส่งภาพ 1 ภาพ → JSON เหตุ + needs_review — ห่อ llama-server (GGUF Q4_K_M) บนเครื่อง
รัน: api/run.sh (เปิด llama-server + uvicorn) · ดู README "HTTP API" สำหรับ key ของคำตอบและตัวแปร env

ขั้นตอน: ตรวจไฟล์ (≤10 MB, ≤40 MP) → to_rgb_resized(512) (เหมือนตอนสร้าง dataset) → PNG data URL (lossless) → dvl.prompt.build_messages
→ llama-server (ทีละคำขอ เพราะเปิด -np 1) → parse_output → กติกา abstain (no_incident ที่ P(เป็นเหตุ) >= T → unsure)"""
import asyncio, hmac, io, json, os, time, warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from dvl.catalog import ALLOWED_TYPES
from dvl.imgutil import to_rgb_resized
from dvl.llamacpp import abstain, gate_fires, p_incident, png_data_url, parse_response, request_body, \
    response_logprobs, value_alternatives
from dvl.prompt import build_messages
from dvl.schema import parse_output, to_api

MAX_BYTES = 10 * 1024 * 1024
MAX_BODY = MAX_BYTES + 64 * 1024  # เผื่อ overhead ของ multipart
MAX_PIXELS = 40_000_000  # กัน decompression bomb: PNG 1-bit ไม่กี่ร้อย byte ประกาศ 20000×20000 ได้
Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # PIL raise แทน warn (เกิน 2× จะ raise DecompressionBombError)
# 1×–2× cap PIL แค่ warn — _decode ตรวจ w*h เองแล้ว จึงปิด warning นี้ (เกิน 2× PIL raise → 413)
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
DECODE_CONCURRENCY = 2  # ถอดภาพพร้อมกันได้กี่ภาพ (CPU/RAM)
ALLOWED_MIME = {"image/jpeg": "JPEG", "image/jpg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
DEFAULT_T = 3.2e-5  # เลือกบน val สำหรับ Q4_K_M + llama.cpp b10909 (reports/abstain-q4.md)


@dataclass(frozen=True)
class Settings:
    llama_url: str = "http://127.0.0.1:8091"
    abstain_t: float = DEFAULT_T
    model_name: str = "dvl-qwen3.5-2b-Q4_K_M"
    api_key: str | None = None
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        return cls(llama_url=e.get("DVL_LLAMA_URL", cls.llama_url).rstrip("/"),
                   abstain_t=float(e.get("DVL_ABSTAIN_T", cls.abstain_t)),
                   model_name=e.get("DVL_MODEL_NAME", cls.model_name),
                   api_key=e.get("DVL_API_KEY") or None,
                   timeout=float(e.get("DVL_TIMEOUT", cls.timeout)))


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str):
        self.status, self.error, self.message = status, error, message


def _decode(data: bytes) -> Image.Image:
    """bytes → ภาพ RGB ≤512 — ApiError ถ้าไม่ใช่ jpeg/png/webp, เกิน MAX_PIXELS หรือถอดรหัสไม่ได้"""
    try:
        img = Image.open(io.BytesIO(data))  # อ่านแค่ header
        fmt = img.format
        if fmt not in ALLOWED_MIME.values():
            raise ApiError(415, "unsupported_media_type", f"image format {fmt} not allowed (jpeg/png/webp)")
        w, h = img.size
        if w * h > MAX_PIXELS:
            raise ApiError(413, "too_many_pixels", f"image {w}x{h} exceeds {MAX_PIXELS} pixels")
        if fmt == "JPEG":
            img.draft("RGB", (1024, 1024))  # ถอดแบบย่อ 1/2–1/8 ตั้งแต่ DCT (ยังใหญ่กว่า 512 ให้ LANCZOS ย่อต่อ)
        img.load()
        return to_rgb_resized(img, max_side=512)
    except ApiError:
        raise
    except Image.DecompressionBombError as ex:
        raise ApiError(413, "too_many_pixels", f"image exceeds {MAX_PIXELS} pixels") from ex
    except Exception as ex:  # PIL โยนได้หลายแบบ (UnidentifiedImageError, OSError, SyntaxError, ...)
        raise ApiError(400, "undecodable_image", f"cannot decode image: {type(ex).__name__}") from ex


class BodyLimit:
    """ASGI middleware: จำกัดขนาด body ทั้งจาก Content-Length และจากการนับ byte ระหว่างอ่าน (chunked ไม่มี header)
    เกิน → 413 too_large และหยุดอ่าน body ทันที (multipart parser ไม่ได้ spool ต่อ)"""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def _reject(self, send):
        body = json.dumps({"error": "too_large", "message": f"image larger than {MAX_BYTES} bytes"},
                          separators=(",", ":")).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = dict(scope["headers"]).get(b"content-length", b"")
        if size.isdigit() and int(size) > self.limit:
            return await self._reject(send)
        seen, over, started = 0, False, False

        async def limited_receive():
            nonlocal seen, over
            if over:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.limit:
                    over = True
                    return {"type": "http.disconnect"}  # แอปเลิกอ่าน (ClientDisconnect)
            return msg

        async def guarded_send(msg):
            nonlocal started
            if over:
                return  # ทิ้งคำตอบของแอป (เช่น 400 parse error) — จะส่ง 413 แทน
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not over:
                raise
        if over and not started:
            await self._reject(send)


def classify_result(resp: dict, t: float) -> dict:
    """คำตอบ llama-server → ฟิลด์ผลลัพธ์ (ยกเว้น model/threshold/latency_ms)"""
    text, conf, _ = parse_response(resp)
    p = parse_output(text)
    pred = {**to_api(p, conf if conf is not None else 0.0),
            "category_alts": value_alternatives(response_logprobs(resp), "category")}
    fired = gate_fires(pred, t)
    final = abstain(pred, t)
    if not p.valid:
        reason = "invalid_output"
    elif fired:
        reason = "uncertain_no_incident"
    elif final["incident_type"] == "unsure":
        reason = "model_unsure"
    else:
        reason = None
    return {"category": final["category"], "incident_type": final["incident_type"],
            "model_incident_type": p.incident_type if p.valid else None, "severity": final["severity"],
            "confidence": final["confidence"], "p_incident": round(p_incident(pred), 8),
            "needs_review": reason is not None, "review_reason": reason,
            "incident_type_name_th": ALLOWED_TYPES.get(final["incident_type"], (None, None))[1], "valid": p.valid}


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    state: dict = {"client": None}
    lock = asyncio.Lock()  # llama-server เปิด -np 1 → ส่งทีละคำขอ
    decode_slots = asyncio.Semaphore(DECODE_CONCURRENCY)

    def llama() -> httpx.AsyncClient:
        if state["client"] is None:
            state["client"] = httpx.AsyncClient(base_url=s.llama_url, transport=transport, timeout=s.timeout)
        return state["client"]

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if state["client"] is not None:
            await state["client"].aclose()

    app = FastAPI(title="disaster-vlm-lab API", version="1", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(_req: Request, ex: ApiError):
        return JSONResponse({"error": ex.error, "message": ex.message}, status_code=ex.status)

    app.add_middleware(BodyLimit, limit=MAX_BODY)

    def require_key(request: Request) -> None:
        if s.api_key and not hmac.compare_digest(request.headers.get("x-api-key", "").encode(), s.api_key.encode()):
            raise ApiError(401, "unauthorized", "missing or wrong X-API-Key")

    @app.get("/health")
    async def health():
        try:
            r = await llama().get("/health", timeout=5.0)
            ok = r.status_code == 200
            detail = "ok" if ok else f"llama-server /health {r.status_code}"
        except httpx.HTTPError as ex:
            ok, detail = False, f"llama-server unreachable: {type(ex).__name__}"
        return JSONResponse({"status": "ok" if ok else "unavailable", "llama": detail, "model": s.model_name},
                            status_code=200 if ok else 503)

    @app.get("/v1/labels", dependencies=[Depends(require_key)])
    async def labels():
        return {"labels": [{"incident_type": k, "category": c, "name_th": th} for k, (c, th) in ALLOWED_TYPES.items()]}

    @app.post("/v1/classify", dependencies=[Depends(require_key)])
    async def classify(image: UploadFile | None = None):
        t0 = time.perf_counter()
        if image is None:
            raise ApiError(400, "missing_image", "multipart field 'image' is required")
        ctype = (image.content_type or "").split(";")[0].strip().lower()
        if ctype not in ALLOWED_MIME:
            raise ApiError(415, "unsupported_media_type", f"content type {ctype or '?'} not allowed (jpeg/png/webp)")
        data = await image.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ApiError(413, "too_large", f"image larger than {MAX_BYTES} bytes")
        async with decode_slots:
            url = await asyncio.to_thread(lambda: png_data_url(_decode(data)))
        body = request_body(build_messages(url))
        async with lock:
            try:
                r = await llama().post("/v1/chat/completions", json=body)
            except httpx.TimeoutException as ex:
                raise ApiError(504, "llama_timeout", f"llama-server did not answer within {s.timeout}s") from ex
            except httpx.HTTPError as ex:
                raise ApiError(502, "llama_unreachable", f"llama-server: {type(ex).__name__}") from ex
        if r.status_code != 200:
            raise ApiError(502, "llama_error", f"llama-server returned {r.status_code}")
        try:
            out = classify_result(r.json(), s.abstain_t)
        except (ValueError, KeyError, IndexError, TypeError) as ex:
            raise ApiError(502, "llama_error", f"bad llama-server response: {type(ex).__name__}") from ex
        return {**out, "model": s.model_name, "threshold": s.abstain_t,
                "latency_ms": round((time.perf_counter() - t0) * 1000)}

    return app


app = create_app()

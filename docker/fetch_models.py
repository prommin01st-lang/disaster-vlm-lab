"""ดาวน์โหลด GGUF ที่ API ต้องใช้ (Q4_K_M + mmproj F16) จาก HF repo private Petanque/dvl-qwen3.5-2b-gguf เมื่อยังไม่มี
ใช้ stdlib ล้วน — service `models` ใน docker-compose.yml รันด้วย python:3.12-slim · รันบนเครื่องก็ได้:
    HF_TOKEN=hf_... python docker/fetch_models.py models/gguf

ไฟล์ที่มีอยู่แล้วและขนาดตรง → ข้าม (ไม่ต้องใช้ token) · ดาวน์โหลดลง <ชื่อ>.part แล้วตรวจ sha256 ก่อน rename"""
import hashlib, os, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

REPO = os.environ.get("DVL_HF_MODEL_REPO", "Petanque/dvl-qwen3.5-2b-gguf")
REVISION = os.environ.get("DVL_HF_MODEL_REVISION", "main")
# (ชื่อไฟล์, ขนาด, sha256) — ค่าจาก LFS ของ HF repo; threshold ของ API ผูกกับ Q4_K_M ตัวนี้
FILES = [
    ("dvl-qwen3.5-2b-Q4_K_M.gguf", 1274396096, "fc54eae31a6bbef529754efd1aaf9aa59662194bb4b96a05adeb8c872cc34d98"),
    ("mmproj-dvl-qwen3.5-2b-F16.gguf", 668226592, "77d7cb2a85629bfeee146d43be07cc54394926e9aeb0822cd7a820b360817289"),
]


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """HF redirect ไป CDN (คนละ host) — ห้ามส่ง Authorization ต่อ (token รั่ว + CDN ปฏิเสธ)"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


def download(name: str, size: int, sha256: str, dest: Path, token: str) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    part = dest.with_name(dest.name + ".part")
    h, done, last = hashlib.sha256(), 0, -1
    with urllib.request.build_opener(_DropAuthOnRedirect).open(req, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            pct = done * 100 // size
            if pct // 10 != last // 10:
                print(f"[models] {name}: {pct}%", flush=True)
                last = pct
    if done != size or h.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: got {done} bytes sha256 {h.hexdigest()} — expected {size} / {sha256}")
    part.replace(dest)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DVL_MODELS_DIR", "/models"))
    missing = [(n, s, d) for n, s, d in FILES if not ((out / n).is_file() and (out / n).stat().st_size == s)]
    if not missing:
        print(f"[models] all GGUF files present in {out} — nothing to download")
        return 0
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print(f"[models] missing in {out}: {', '.join(n for n, _, _ in missing)}\n"
              f"[models] set HF_TOKEN (read access to private repo {REPO}) or put the files there yourself", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, size, sha in missing:
        print(f"[models] downloading {REPO}/{name} ({size / 2**30:.2f} GiB) → {out}", flush=True)
        try:
            download(name, size, sha, out / name, token)
        except urllib.error.HTTPError as ex:
            print(f"[models] {name}: HTTP {ex.code} — token without access to {REPO}?", file=sys.stderr)
            return 1
    print("[models] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

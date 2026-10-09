"""สร้างภาพ fixture 4 ไฟล์ของ Bruno collection ที่ไม่แจกใน repo (ภาพ CrisisMMD/Flickr — ลิขสิทธิ์ของเจ้าของภาพ)

ดึงจาก HF dataset private Petanque/dvl-data (dataset_v1.tar) แบบ stream — แตกเฉพาะ 4 ภาพ ไม่เก็บ tar ไว้ (~600 MB ผ่านเน็ต)
หรือคัดลอกจาก dataset ที่มีบนเครื่องแล้ว · ใช้ stdlib ล้วน (python 3.9+) รันจาก clone ใหม่ได้:

    HF_TOKEN=hf_... python bruno/fetch_fixtures.py               # จาก HF (token อ่าน Petanque/dvl-data ได้)
    python bruno/fetch_fixtures.py --from-local data/dataset_v1  # จาก dataset บนเครื่อง

ตรวจ sha256 ทุกไฟล์ แล้วเขียน bruno/.env (gitignored) → DVL_FIXTURES=<ไฟล์ที่มี> ให้ request 01–03 รู้ว่าไม่ต้อง skip
(Bruno sandbox ค่าเริ่มต้นอ่านไฟล์ไม่ได้) · ไม่มี token จะใช้ ~/.cache/huggingface/token ถ้ามี"""
import argparse, hashlib, json, os, shutil, sys, tarfile, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO = "Petanque/dvl-data"
TAR = "dataset_v1.tar"
# fixture → (id ใน test_gold, sha256) — ที่มาดูตาราง Fixtures ใน collection.bru
WANTED = {
    "flood.jpg": ("e374542ee5ccab03", "498a4001acf94cb86425219b3a95bd8575ff30db9fc9ee8122c379d98444b7f7"),
    "building-fire.jpg": ("05567df96c3e618d", "0b0016b5e61b9746d7ee8f8aeb672f657636c43d88e88da84776e6a69fd5f4b6"),
    "no-incident.jpg": ("fb4fa14ff082ee2f", "03b6f8dfe8dcb82a2c4c5ccadc734ea43f282bc3b18fe5c689fd2ec477c5f41b"),
    "uncertain-no-incident.jpg": ("fd5a1ada5004c085", "0f3b2d349a04785d5b7a4f2b452fc9166fce5a53c544d69746ca40e0d1c729d2"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_paths(gold_lines) -> dict[str, str]:
    """แถวของ test_gold.jsonl → {fixture: path ของภาพเทียบกับโฟลเดอร์ dataset (เช่น images/<id>.jpg)}"""
    by_id = {}
    for line in gold_lines:
        if line.strip():
            row = json.loads(line)
            by_id[row["id"]] = row["image"]
    missing = [i for i, _ in WANTED.values() if i not in by_id]
    if missing:
        raise SystemExit(f"[fixtures] ids not in test_gold.jsonl: {missing}")
    return {name: by_id[i] for name, (i, _) in WANTED.items()}


def verify(name: str, path: Path) -> None:
    want = WANTED[name][1]
    got = sha256(path)
    if got != want:
        path.unlink()
        raise SystemExit(f"[fixtures] {name}: sha256 {got} != expected {want}")


def from_local(dataset: Path) -> None:
    if not (dataset / "test_gold.jsonl").is_file():
        raise SystemExit(f"[fixtures] {dataset}/test_gold.jsonl not found")
    paths = image_paths((dataset / "test_gold.jsonl").read_text(encoding="utf-8").splitlines())
    for name, rel in paths.items():
        shutil.copyfile(dataset / rel, FIXTURES / name)
        verify(name, FIXTURES / name)
        print(f"[fixtures] {name} ← {dataset / rel}")


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """HF redirect ไป CDN (คนละ host) — ไม่ส่ง Authorization ต่อ"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


def _open(filename: str, token: str):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return urllib.request.build_opener(_DropAuthOnRedirect).open(req, timeout=60)


def _token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        f = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
        tok = f.read_text().strip() if f.is_file() else ""
    if not tok:
        raise SystemExit(f"[fixtures] set HF_TOKEN (read access to private dataset {REPO}) or use --from-local")
    return tok


def from_hf() -> None:
    token = _token()
    with _open("test_gold.jsonl", token) as r:
        paths = image_paths(r.read().decode("utf-8").splitlines())
    # member ใน tar = dataset_v1/<image path>
    members = {f"dataset_v1/{rel}": name for name, rel in paths.items()}
    print(f"[fixtures] streaming {REPO}/{TAR} (extracting {len(members)} files, nothing else is kept)", flush=True)
    with _open(TAR, token) as r, tarfile.open(fileobj=r, mode="r|") as tar:
        for m in tar:
            name = members.pop(m.name, None)
            if name is None or not m.isfile():
                continue
            src = tar.extractfile(m)
            (FIXTURES / name).write_bytes(src.read())
            verify(name, FIXTURES / name)
            print(f"[fixtures] {name} ← {m.name}", flush=True)
            if not members:
                break  # ครบแล้ว หยุด stream
    if members:
        raise SystemExit(f"[fixtures] not found in {TAR}: {sorted(members)}")


def write_env() -> None:
    """bruno/.env: DVL_FIXTURES = fixture ที่มีอยู่จริงตอนนี้ (Bruno CLI อ่านเป็น process env)"""
    have = [n for n in WANTED if (FIXTURES / n).is_file()]
    env = HERE / ".env"
    lines = [l for l in (env.read_text().splitlines() if env.is_file() else []) if not l.startswith("DVL_FIXTURES=")]
    env.write_text("\n".join(lines + [f"DVL_FIXTURES={','.join(have)}"]) + "\n")
    print(f"[fixtures] {env.relative_to(HERE.parent)}: DVL_FIXTURES={','.join(have) or '(none)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-local", type=Path, metavar="DATASET_DIR",
                    help="โฟลเดอร์ dataset ที่มี test_gold.jsonl + images/ (เช่น data/dataset_v1)")
    ap.add_argument("--force", action="store_true", help="ดึงใหม่แม้มีไฟล์ครบแล้ว")
    args = ap.parse_args()
    FIXTURES.mkdir(exist_ok=True)
    ok = all((FIXTURES / n).is_file() and sha256(FIXTURES / n) == h for n, (_, h) in WANTED.items())
    if ok and not args.force:
        print("[fixtures] all 4 fixtures present (sha256 ok)")
    elif args.from_local:
        from_local(args.from_local)
    else:
        from_hf()
    write_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())

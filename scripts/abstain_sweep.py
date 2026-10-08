"""กติกาตอนใช้งาน: ทายว่า no_incident แต่ P(เป็นเหตุ) >= T → เปลี่ยนเป็น unsure (ส่งให้คนดู) ไม่ต้องเทรนใหม่
P(เป็นเหตุ) = 1 − P(category = null) จาก category_alts (predict_gguf.py) — category มาก่อน incident_type
จึงเป็นจุดที่โมเดลตัดสินว่าเหตุหรือไม่ (confidence ของ incident_type อิ่มที่ 1.0 ใช้แยกไม่ได้)
เลือก T บนชุดหนึ่ง (เช่น val) แล้วเอา T นั้นไปวัดบนชุดอื่น — ห้ามเลือก T บนชุดที่ใช้รายงานผล
ใช้: python scripts/abstain_sweep.py --tune GOLD.jsonl PRED.jsonl [--target-miss 0.01]
                                   [--check GOLD.jsonl PRED.jsonl [--exclude GOLD.jsonl]] ...
--exclude ตัดแถวที่มี id อยู่ในไฟล์นั้นออกจากทุกชุด ยกเว้นชุดที่เป็นไฟล์นั้นเอง (เช่น test ลบ test_gold)"""
import argparse, json, math

from dvl.metrics import compute_metrics

GRID = [10 ** (-k / 4) for k in range(0, 33)]  # 1 … 1e-8 แบบ log


def load(path: str) -> dict:
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}


def p_incident(p: dict) -> float:
    p_null = sum(math.exp(lp) for tok, lp in p["category_alts"] if tok.strip() == "null")
    return max(0.0, 1.0 - p_null)


def abstain(p: dict, t: float) -> dict:
    if t is not None and p["incident_type"] == "no_incident" and p_incident(p) >= t:
        return {**p, "category": "other", "incident_type": "unsure", "severity": "mild"}
    return p


def score(gold: dict, pred: dict, t: float) -> dict:
    ids = [i for i in gold if i in pred]
    m = compute_metrics([gold[i] for i in ids], [abstain(pred[i], t) for i in ids])
    return {k: m[k] for k in ("n", "type_macro_f1", "false_alarm_rate", "miss_rate", "unsure_rate")}


def row(name: str, t, m: dict) -> str:
    return (f"| {name} | {t} | {m['n']} | {m['type_macro_f1']:.3f} | {m['false_alarm_rate']:.3f} | "
            f"{m['miss_rate']:.3f} | {m['unsure_rate']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", nargs=2, required=True, metavar=("GOLD", "PRED"))
    ap.add_argument("--target-miss", type=float, default=0.01)
    ap.add_argument("--check", nargs=2, action="append", default=[], metavar=("GOLD", "PRED"))
    ap.add_argument("--exclude", action="append", default=[])
    a = ap.parse_args()

    tg, tp = load(a.tune[0]), load(a.tune[1])
    print("sweep on", a.tune[1])
    print("| T | n | macro-F1 | false_alarm | miss | unsure(review) |\n|---|---|---|---|---|---|")
    chosen = None
    for t in [None] + GRID:
        m = score(tg, tp, t)
        print(row("", "off" if t is None else f"{t:.1e}", m)[3:])
        if t is not None and chosen is None and m["miss_rate"] <= a.target_miss:
            chosen = t  # T ใหญ่สุด (= ส่งให้คนดูน้อยสุด) ที่ miss ถึงเป้า
    if chosen is None:
        raise SystemExit(f"no T reaches miss <= {a.target_miss}")
    print(f"\nchosen T = {chosen:.1e} (largest T with tune-set miss <= {a.target_miss})\n")

    excl = set()
    for path in a.exclude:
        excl |= set(load(path))
    print("| set | T | n | macro-F1 | false_alarm | miss | unsure(review) |\n|---|---|---|---|---|---|---|")
    for gpath, ppath in [tuple(a.tune)] + [tuple(c) for c in a.check]:
        g = load(gpath)
        if gpath not in a.exclude:
            g = {k: v for k, v in g.items() if k not in excl}
        name = f"{ppath.split('/')[-1].removeprefix('predictions-').removesuffix('.jsonl')} on {gpath.split('/')[-1]}"
        print(row(name, "off", score(g, load(ppath), None)))
        print(row(name, f"{chosen:.1e}", score(g, load(ppath), chosen)))


if __name__ == "__main__":
    main()

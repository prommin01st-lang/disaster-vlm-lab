from collections import Counter

NOT_ALARM = {"no_incident", "unsure"}


def _ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    n, total = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return round(total, 6)


def compute_metrics(gold: list[dict], pred: list[dict]) -> dict:
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold/pred must be same non-zero length")
    n = len(gold)
    gt = [x["incident_type"] for x in gold]
    pt = [x["incident_type"] for x in pred]
    correct = [a == b for a, b in zip(gt, pt)]

    per_type = {}
    for t in sorted(set(gt)):
        tp = sum(a == t and b == t for a, b in zip(gt, pt))
        fp = sum(a != t and b == t for a, b in zip(gt, pt))
        fn = sum(a == t and b != t for a, b in zip(gt, pt))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_type[t] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    neg = [i for i, t in enumerate(gt) if t == "no_incident"]
    pos = [i for i, t in enumerate(gt) if t != "no_incident"]
    return {
        "n": n,
        "json_valid_rate": sum(bool(x.get("valid", True)) for x in pred) / n,
        "type_accuracy": sum(correct) / n,
        "type_macro_f1": sum(v["f1"] for v in per_type.values()) / len(per_type),
        "category_accuracy": sum(a["category"] == b["category"] for a, b in zip(gold, pred)) / n,
        "severity_accuracy": sum(a["severity"] == b["severity"] for a, b in zip(gold, pred)) / n,
        "false_alarm_rate": sum(pt[i] not in NOT_ALARM for i in neg) / len(neg) if neg else 0.0,
        "miss_rate": sum(pt[i] == "no_incident" for i in pos) / len(pos) if pos else 0.0,
        "unsure_rate": Counter(pt)["unsure"] / n,
        "ece": _ece([float(x.get("confidence", 0.0)) for x in pred], correct),
        "per_type": per_type,
    }

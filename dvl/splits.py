import random
from collections import defaultdict

from dvl.imgutil import hamming


def group_near_duplicates(hashes: list[int], max_dist: int = 4) -> list[int]:
    """union-find แบบ O(n^2) ผ่าน bucket 16 บิตแรก 4 ชุด (LSH แบบง่าย) — ภาพซ้ำ/เกือบซ้ำได้ group เดียวกัน"""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for band in range(4):
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> (16 * band)) & 0xFFFF].append(i)
        for idx in buckets.values():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if hamming(hashes[i], hashes[j]) <= max_dist:
                        parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def stratified_split(labels: list[str], groups: list[int], fractions: dict[str, float], seed: int) -> list[str]:
    """แบ่งตาม group (label ของ group = label ของสมาชิกตัวแรก) แยกตามคลาส"""
    rng = random.Random(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    by_label: dict[str, list[int]] = defaultdict(list)
    for g, idx in members.items():
        by_label[labels[idx[0]]].append(g)
    names = list(fractions)
    out = [""] * len(labels)
    for lab in sorted(by_label):
        gs = sorted(by_label[lab])
        rng.shuffle(gs)
        total = sum(len(members[g]) for g in gs)
        # เติม split ที่เล็กที่สุดก่อน (ไม่ใช่ train) ให้ได้ตามสัดส่วน ที่เหลือเป็น split แรก
        targets = {n: round(total * fractions[n]) for n in names[1:]}
        filled = {n: 0 for n in names}
        for g in gs:
            size = len(members[g])
            dest = next((n for n in names[1:] if filled[n] + size <= targets[n]), names[0])
            filled[dest] += size
            for i in members[g]:
                out[i] = dest
    return out


def balance(rows: list[dict], key: str, cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    out = []
    for k in sorted(by):
        rs = by[k][:]
        rng.shuffle(rs)
        out.extend(rs[:cap])
    return out

import random
from collections import defaultdict

import numpy as np


def group_near_duplicates(hashes: list[int], max_dist: int = 4) -> list[int]:
    """Exact all-pairs Hamming distance check with union-find. Groups hashes within max_dist
    by computing exact Hamming distances in chunked rows; handles large inputs by processing
    in chunks of 1024 to avoid memory blowup."""
    if not hashes:
        return []

    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = np.array(hashes, dtype=np.uint64)
    n = len(hashes)
    chunk_size = 1024

    # Check within-chunk pairs first
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        for i in range(i_start, i_end):
            for j in range(i + 1, i_end):
                d = (h[i] ^ h[j]).bit_count()
                if d <= max_dist:
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj

    # Check cross-chunk pairs (i in chunk vs j after chunk)
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk = h[i_start:i_end]
        if i_end < n:
            dists = np.bitwise_count(chunk[:, None] ^ h[i_end:])
            for chunk_i, dists_row in enumerate(dists):
                i = i_start + chunk_i
                for j_offset, d in enumerate(dists_row):
                    j = i_end + j_offset
                    if d <= max_dist:
                        pi, pj = find(i), find(j)
                        if pi != pj:
                            parent[pi] = pj

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

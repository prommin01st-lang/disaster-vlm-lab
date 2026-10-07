import random
import time
from collections import Counter

from dvl.splits import balance, group_near_duplicates, stratified_split


def test_group_near_duplicates():
    hashes = [0b0, 0b1, 0xFFFF_FFFF_FFFF_FFFF, 0b11]
    g = group_near_duplicates(hashes, max_dist=2)
    assert g[0] == g[1] == g[3] and g[2] != g[0]


def test_split_keeps_groups_together():
    labels = ["a"] * 50 + ["b"] * 50
    groups = [i // 2 for i in range(100)]  # คู่ละ group
    split = stratified_split(labels, groups, {"train": 0.8, "test": 0.2}, seed=1)
    for i in range(0, 100, 2):
        assert split[i] == split[i + 1]


def test_split_stratified_fractions():
    labels = ["a"] * 100 + ["b"] * 100
    split = stratified_split(labels, list(range(200)), {"train": 0.8, "val": 0.05, "test": 0.15}, seed=1)
    for lab in "ab":
        c = Counter(s for s, l in zip(split, labels) if l == lab)
        assert c["test"] == 15 and c["val"] == 5 and c["train"] == 80


def test_split_deterministic():
    labels = ["a"] * 30
    f = {"train": 0.5, "test": 0.5}
    assert stratified_split(labels, list(range(30)), f, 7) == stratified_split(labels, list(range(30)), f, 7)


def test_balance_caps_each_class():
    rows = [{"t": "a"}] * 10 + [{"t": "b"}] * 3
    out = balance(rows, "t", cap=5, seed=1)
    assert Counter(r["t"] for r in out) == {"a": 5, "b": 3}


def test_group_empty_input():
    """Empty input should return empty list."""
    assert group_near_duplicates([]) == []


def test_group_distance_4_exact():
    """Distance-4 pair with one differing bit in each 16-bit band must group at max_dist=4.

    0 = 0b0000...0000
    x = 0b0001_0001_0001_0001 (one bit set in each 16-bit band: bits 0, 16, 32, 48)
    Hamming distance = 4, so they should be grouped at max_dist=4.
    """
    a = 0b0
    b = (1 | (1 << 16) | (1 << 32) | (1 << 48))
    assert (a ^ b).bit_count() == 4
    g = group_near_duplicates([a, b], max_dist=4)
    assert g[0] == g[1], f"Distance-4 pair should group at max_dist=4; got groups {g[0]} vs {g[1]}"


def test_group_distance_5_not_grouped():
    """Distance-5 pair must NOT be grouped at max_dist=4."""
    a = 0b0
    # Create a distance-5 hash by flipping 5 bits
    b = (1 | (1 << 8) | (1 << 16) | (1 << 24) | (1 << 32))
    assert (a ^ b).bit_count() == 5
    g = group_near_duplicates([a, b], max_dist=4)
    assert g[0] != g[1], f"Distance-5 pair should NOT group at max_dist=4; got same group {g[0]}"


def test_group_transitive_closure():
    """Transitive chain a~b~c (where a-c might be >max_dist) should all end up in one group."""
    # a = 0, b = distance 2 from a, c = distance 2 from b, distance 4 from a
    a = 0b0
    b = 0b11  # Distance 2 from a
    c = 0b11 | (1 << 16)  # Distance 2 from b, distance 3 from a; all should group

    g = group_near_duplicates([a, b, c], max_dist=2)
    assert g[0] == g[1] == g[2], f"Transitive chain should all be in one group; got {g}"


def test_group_performance_5000_hashes():
    """5000 random 64-bit hashes should group in under ~10 seconds."""
    rng = random.Random(42)
    hashes = [rng.getrandbits(64) for _ in range(5000)]

    start = time.time()
    result = group_near_duplicates(hashes, max_dist=4)
    elapsed = time.time() - start

    assert len(result) == 5000, f"Should return group id for each hash"
    assert elapsed < 10, f"Grouping 5000 hashes took {elapsed:.2f}s, expected <10s"

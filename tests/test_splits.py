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

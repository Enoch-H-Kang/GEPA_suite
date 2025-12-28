# hover_data.py
from __future__ import annotations

import random
from datasets import load_dataset
import dspy


def _extract_gold_titles(x) -> list[str]:
    # supporting_facts: List[{"key": title, "value": sent_id}]
    sf = x.get("supporting_facts", [])
    titles = []
    for item in sf:
        if isinstance(item, dict):
            t = item.get("key")
            if isinstance(t, str) and t.strip():
                titles.append(t.strip())

    # de-dupe, keep order
    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_hover_splits(seed: int = 0, n_train=150, n_dev=300, n_test=300, *, require_num_hops: int | None = 3):
    """
    Script-free loader that works with modern datasets versions that reject dataset scripts.

    Uses vincentkoc/hover-parquet split=train as a labeled pool, then deterministically
    shuffles and slices into train/dev/test sizes used in GEPA (150/300/300).
    """
    ds = load_dataset("vincentkoc/hover-parquet", split="train")

    pool = []
    seen_hpqa = set()
    for x in ds:
        if require_num_hops is not None and x.get("num_hops") != require_num_hops:
            continue

        hid = x.get("hpqa_id")
        if hid and hid in seen_hpqa:
            continue
        if hid:
            seen_hpqa.add(hid)

        titles = _extract_gold_titles(x)
        if not titles:
            continue

        pool.append(
            dspy.Example(
                claim=x["claim"],
                titles=titles,
                hpqa_id=hid,
                num_hops=x.get("num_hops"),
            ).with_inputs("claim")
        )

    rng = random.Random(seed)
    rng.shuffle(pool)

    need = n_train + n_dev + n_test
    assert len(pool) >= need, f"Not enough usable examples: have {len(pool)}, need {need}"

    train = pool[:n_train]
    dev   = pool[n_train:n_train+n_dev]
    test  = pool[n_train+n_dev:n_train+n_dev+n_test]
    return train, dev, test

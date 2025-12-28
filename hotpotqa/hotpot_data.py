# hotpot_data.py
from __future__ import annotations

import random
from typing import Optional

import dspy
from datasets import load_dataset


def _extract_gold_titles(x) -> list[str]:
    """
    HotpotQA (HF) stores supporting_facts as a dict like:
      { "title": [...], "sent_id": [...] }
    but we make this robust to other shapes.
    """
    sf = x.get("supporting_facts", None)
    titles: list[str] = []

    if isinstance(sf, dict):
        ts = sf.get("title") or sf.get("titles") or sf.get("key")
        if isinstance(ts, list):
            for t in ts:
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())

    elif isinstance(sf, list):
        # sometimes it can be list[dict] or list[tuple/list]
        for item in sf:
            if isinstance(item, dict):
                t = item.get("title") or item.get("key")
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())
            elif isinstance(item, (list, tuple)) and len(item) >= 1:
                t = item[0]
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())

    # de-dupe, keep order
    seen = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_hotpot_splits(
    seed: int = 0,
    n_train: int = 150,
    n_dev: int = 300,
    n_test: int = 300,
    *,
    subset: str = "fullwiki",
    pool_split: str = "train",
    require_type: Optional[str] = None,   # e.g. "bridge" or "comparison"
    require_level: Optional[str] = None,  # e.g. "easy" / "medium" / "hard"
):
    """
    Deterministically shuffle and slice a labeled pool into (train/dev/test).

    - Uses hotpotqa/hotpot_qa which is parquet-converted on HF.
    - We sample from one labeled pool split (default: train) to create
      our 150/300/300 splits like in GEPA.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", subset, split=pool_split)

    pool = []
    seen_ids = set()

    for x in ds:
        ex_id = x.get("id")
        if ex_id and ex_id in seen_ids:
            continue
        if ex_id:
            seen_ids.add(ex_id)

        if require_type is not None and x.get("type") != require_type:
            continue
        if require_level is not None and x.get("level") != require_level:
            continue

        q = x.get("question")
        a = x.get("answer")
        if not (isinstance(q, str) and q.strip()):
            continue
        if not (isinstance(a, str) and a.strip()):
            continue

        titles = _extract_gold_titles(x)
        if not titles:
            continue

        pool.append(
            dspy.Example(
                question=q.strip(),
                answer=a.strip(),
                titles=titles,
                id=ex_id,
                qtype=x.get("type"),
                level=x.get("level"),
            ).with_inputs("question")
        )

    rng = random.Random(seed)
    rng.shuffle(pool)

    need = n_train + n_dev + n_test
    assert len(pool) >= need, f"Not enough usable examples: have {len(pool)}, need {need}"

    train = pool[:n_train]
    dev = pool[n_train:n_train + n_dev]
    test = pool[n_train + n_dev:n_train + n_dev + n_test]
    return train, dev, test

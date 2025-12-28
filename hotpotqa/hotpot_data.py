# hotpot_data.py
from __future__ import annotations

import random
from typing import Any

import dspy
from datasets import load_dataset


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_gold_titles(x: dict[str, Any]) -> list[str]:
    """
    HotpotQA gold supporting docs come from supporting_facts.

    Robustly supports:
      1) HF columnar form:
         supporting_facts = {"title": [...], "sent_id": [...]}
      2) canonical json form:
         supporting_facts = [[title, sent_id], ...]
      3) parquet-ish conversions:
         supporting_facts = [{"title": ...}, ...] or [{"key": ...}, ...]
    """
    sf = x.get("supporting_facts") or []
    titles: list[str] = []

    if isinstance(sf, dict):
        tlist = sf.get("title") or sf.get("titles") or []
        if isinstance(tlist, list):
            for t in tlist:
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())

    elif isinstance(sf, list):
        for item in sf:
            t = None
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                t = item[0]
            elif isinstance(item, dict):
                t = item.get("title") or item.get("key") or item.get("wiki_title")
            if isinstance(t, str) and t.strip():
                titles.append(t.strip())

    return _dedupe_keep_order(titles)


def load_hotpotqa_splits(
    seed: int = 0,
    n_train: int = 150,
    n_dev: int = 300,
    n_test: int = 300,
    *,
    dataset_name: str = "hotpotqa/hotpot_qa",
    config: str = "fullwiki",
    pool_split: str = "train",
    require_answer: bool = True,
):
    """
    Deterministically sample 150/300/300 from a labeled pool.

    GEPA paper specifies the sizes but not which official split is used;
    the simplest (and fully-labeled) approach is to sample from `train`. :contentReference[oaicite:5]{index=5}

    HF HotpotQA supports configs like "fullwiki". :contentReference[oaicite:6]{index=6}
    """
    ds = load_dataset(dataset_name, config, split=pool_split)

    pool: list[dspy.Example] = []
    seen_ids: set[str] = set()

    for x in ds:
        q = x.get("question")
        a = x.get("answer")
        _id = x.get("id")

        if not isinstance(q, str) or not q.strip():
            continue

        if require_answer and (not isinstance(a, str) or not a.strip()):
            continue

        titles = _extract_gold_titles(x)
        if not titles:
            continue

        if isinstance(_id, str) and _id:
            if _id in seen_ids:
                continue
            seen_ids.add(_id)

        pool.append(
            dspy.Example(
                question=q.strip(),
                answer=(a.strip() if isinstance(a, str) else ""),
                titles=titles,
                id=_id,
                type=x.get("type"),
                level=x.get("level"),
            ).with_inputs("question")
        )

    rng = random.Random(seed)
    rng.shuffle(pool)

    need = n_train + n_dev + n_test
    if len(pool) < need:
        raise ValueError(f"Not enough usable examples: have {len(pool)}, need {need}")

    train = pool[:n_train]
    dev = pool[n_train:n_train + n_dev]
    test = pool[n_train + n_dev:n_train + n_dev + n_test]
    return train, dev, test

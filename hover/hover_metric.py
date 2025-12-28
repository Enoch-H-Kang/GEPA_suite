# hover_metric.py
def normalize_title(t: str) -> str:
    return t.strip()

def hover_recall(example, pred, trace=None):
    gold = {normalize_title(t) for t in example.titles}
    got  = {normalize_title(t) for t in pred.titles}
    recall = len(gold & got) / max(1, len(gold))

    # For optimizers that expect bool "success" during compilation,
    # return True only when perfect recall. This pattern is used in DSPy examples. 
    if trace is not None:
        return recall >= 1.0
    return recall

def hover_feedback_text(example, pred) -> str:
    gold = [normalize_title(t) for t in example.titles]
    got  = {normalize_title(t) for t in pred.titles}

    correct = [t for t in gold if t in got]
    missing = [t for t in gold if t not in got]

    # “Simply identifies the set ... and returns them as feedback text.” :contentReference[oaicite:12]{index=12}
    return f"Correct gold docs retrieved: {correct}\nGold docs still missing: {missing}"

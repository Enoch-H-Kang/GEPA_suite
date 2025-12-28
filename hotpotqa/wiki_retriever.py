# wiki_retriever.py
from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Callable, List, Tuple

import ujson
import bm25s
import Stemmer
from dspy.utils import download  # DSPy tutorial uses this exact helper. :contentReference[oaicite:1]{index=1}


WIKI_TAR_URL = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
WIKI_TAR_NAME = "wiki.abstracts.2017.tar.gz"
WIKI_JSONL_NAME = "wiki.abstracts.2017.jsonl"


def ensure_wiki_abstracts_2017(data_dir: str | Path) -> Path:
    """
    Ensures wiki.abstracts.2017.tar.gz is downloaded and extracted into data_dir,
    producing wiki.abstracts.2017.jsonl.

    Follows DSPy tutorial:
      from dspy.utils import download
      download(".../wiki.abstracts.2017.tar.gz")
      tar -xzvf wiki.abstracts.2017.tar.gz
    :contentReference[oaicite:2]{index=2}
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / WIKI_TAR_NAME
    jsonl_path = data_dir / WIKI_JSONL_NAME

    # If already extracted, we're done.
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return jsonl_path

    # Download tarball if needed.
    if not tar_path.exists() or tar_path.stat().st_size == 0:
        # dspy.utils.download downloads into CWD by default; we control cwd via chdir.
        cwd = os.getcwd()
        try:
            os.chdir(data_dir)
            download(WIKI_TAR_URL)  # creates wiki.abstracts.2017.tar.gz in data_dir
        finally:
            os.chdir(cwd)

    if not tar_path.exists():
        raise FileNotFoundError(f"Expected {tar_path} after download, but it does not exist.")

    # Extract tarball.
    # Some archives contain a single jsonl at root; extract all, then check presence.
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    if not jsonl_path.exists():
        # Sometimes the tar extracts to current dir; double-check common alt locations.
        candidates = list(data_dir.rglob(WIKI_JSONL_NAME))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            f"Extracted {tar_path} but did not find {WIKI_JSONL_NAME} under {data_dir}."
        )

    return jsonl_path


def load_wiki_corpus(jsonl_path: str | Path) -> List[str]:
    """
    Loads wiki.abstracts.2017.jsonl into a list[str] where each entry is:
        "{title} | {' '.join(text)}"
    Exactly as in the DSPy multihop retrieval tutorial. :contentReference[oaicite:3]{index=3}
    """
    jsonl_path = Path(jsonl_path)
    corpus: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = ujson.loads(line)
            title = obj["title"]
            text = obj["text"]
            # In tutorial: ' '.join(line['text'])
            corpus.append(f"{title} | {' '.join(text)}")

    return corpus


def build_or_load_bm25(
    *,
    wiki_dir: str | Path,
    index_dir: str | Path,
    k1: float = 0.9,
    b: float = 0.4,
) -> tuple[List[str], bm25s.BM25, Stemmer.Stemmer]:
    """
    Builds or loads a BM25 index over the wiki abstracts 2017 corpus.
    We persist both:
      - the corpus list (documents.jsonl)
      - the BM25 index directory

    Note: bm25s has built-in save/load.
    """
    wiki_dir = Path(wiki_dir)
    index_dir = Path(index_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    docs_path = index_dir / "documents.jsonl"
    bm25_path = index_dir / "bm25_index"  # bm25s will create files in this folder

    # If already cached, load corpus + index.
    if docs_path.exists() and bm25_path.exists():
        corpus: List[str] = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                corpus.append(ujson.loads(line)["doc"])
        retriever = bm25s.BM25.load(str(bm25_path))
        stemmer = Stemmer.Stemmer("english")
        return corpus, retriever, stemmer

    # Otherwise download/extract + build.
    jsonl_path = ensure_wiki_abstracts_2017(wiki_dir)
    corpus = load_wiki_corpus(jsonl_path)

    # Save corpus for reuse
    with docs_path.open("w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(ujson.dumps({"doc": doc}) + "\n")

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens)
    retriever.save(str(bm25_path))

    return corpus, retriever, stemmer


def make_search_fn(corpus, retriever, stemmer, *, n_threads: int = 1):
    def search(query: str, k: int = 5):
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
        results, scores = retriever.retrieve(tokens, k=k, n_threads=n_threads, show_progress=False)
        docs = [corpus[i] for i in results[0]]
        scs  = [float(s) for s in scores[0]]
        return docs, scs
    return search






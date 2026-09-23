"""Segments -> candidate primitives.

Pipeline: represent -> embed -> reduce -> HDBSCAN -> c-TF-IDF naming. No LLM anywhere in the
extraction, so the operations that come out are a property of the corpus rather than of a model
asked to invent them.

Two dials matter:
  min_cluster_size      how many operations you get
  cluster_selection     "eom" merges toward broader clusters, "leaf" keeps finer ones

Neither has a universally right value; `sweep()` reports what each setting yields on YOUR traces
and `stability()` says which settings survive a split-half rerun.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .segment import Segment

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Cluster:
    id: int
    size: int
    terms: list[str]                       # c-TF-IDF, most cluster-specific first
    exemplars: list[str] = field(default_factory=list)   # medoid segments, masked
    exemplar_raw: list[str] = field(default_factory=list)
    lift: float | None = None              # validator lift vs corpus base rate
    base_rate: float | None = None
    label_rate: float | None = None

    @property
    def name(self) -> str:
        """A slug from the top terms. Deliberately mechanical -- rename by hand."""
        return "_".join(self.terms[:2]) if self.terms else f"cluster_{self.id}"


@dataclass
class ClusterResult:
    clusters: list[Cluster]
    labels: list[int]
    n_noise: int
    params: dict

    def to_json(self) -> str:
        return json.dumps(
            {"params": self.params, "n_noise": self.n_noise,
             "clusters": [asdict(c) | {"name": c.name} for c in self.clusters]},
            indent=1, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def embed(texts: Sequence[str], model_name: str = ST_MODEL, batch: int = 256) -> np.ndarray:
    """Sentence embeddings, L2-normalised.

    Falls back to a deterministic character n-gram TF-IDF projection when sentence-transformers
    is not installed. The fallback keeps the pipeline runnable (and the tests offline) but it is
    a weaker representation -- install the extra for real runs.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return _tfidf_embed(texts)
    m = SentenceTransformer(model_name)
    return np.asarray(m.encode(list(texts), batch_size=batch, show_progress_bar=False,
                               normalize_embeddings=True, convert_to_numpy=True))


def _tfidf_embed(texts: Sequence[str], dims: int = 128) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = V.fit_transform(list(texts))
    k = int(min(dims, max(2, min(X.shape) - 1)))
    Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def reduce_dims(X: np.ndarray, n: int = 10, seed: int = 0) -> np.ndarray:
    if X.shape[0] <= n + 2:
        return X
    try:
        import umap
        return umap.UMAP(n_components=n, n_neighbors=15, min_dist=0.0,
                         metric="cosine", random_state=seed).fit_transform(X)
    except Exception:
        # PCA keeps the pipeline runnable without numba; density clustering is weaker in the
        # linear projection, so this is a fallback, not a default.
        from sklearn.decomposition import PCA
        return PCA(n_components=min(n, X.shape[1]), random_state=seed).fit_transform(X)


def hdbscan(Z: np.ndarray, min_size: int = 20, min_samples: int = 10,
            selection: str = "eom") -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=min_size, min_samples=min_samples,
                   cluster_selection_method=selection, metric="euclidean",
                   copy=True).fit_predict(Z)


def ctfidf(docs_by_cluster: dict[int, list[str]], top_k: int = 8,
           max_features: int = 20000) -> dict[int, list[str]]:
    """Class-based TF-IDF: rank terms by how specific they are to a cluster.

    w = tf(t, c) * log(1 + A / f(t)), with A the mean tokens per class.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    labels = sorted(docs_by_cluster)
    corpus = [" ".join(docs_by_cluster[k]) for k in labels]
    cv = CountVectorizer(max_features=max_features, stop_words="english",
                         token_pattern=r"(?u)\b\w[\w_]+\b")
    tf = cv.fit_transform(corpus).toarray().astype(float)
    terms = np.array(cv.get_feature_names_out())
    lengths = tf.sum(axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    f_t = tf.sum(axis=0)
    f_t[f_t == 0] = 1.0
    w = (tf / lengths) * np.log(1.0 + tf.sum() / max(tf.shape[0], 1) / f_t)
    out = {}
    for i, k in enumerate(labels):
        idx = np.argsort(-w[i])[:top_k]
        out[k] = [str(t) for t in terms[idx] if w[i][cv.vocabulary_[t]] > 0]
    return out


def _medoids(X: np.ndarray, idx: np.ndarray, k: int = 3) -> list[int]:
    """The k members closest to the cluster centroid -- the most typical, not the most extreme."""
    c = X[idx].mean(axis=0)
    d = np.linalg.norm(X[idx] - c, axis=1)
    return [int(idx[i]) for i in np.argsort(d)[:k]]


def centroids(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for k in sorted(set(labels.tolist())):
        if k == -1:
            continue
        v = X[labels == k].mean(axis=0)
        n = float(np.linalg.norm(v))
        out[k] = v / n if n else v
    return out


def match_centroids(a: dict[int, np.ndarray], b: dict[int, np.ndarray],
                    thresh: float = 0.80) -> float:
    """Fraction of a's clusters with a partner in b above cosine `thresh`."""
    if not a or not b:
        return 0.0
    B = np.stack([b[k] for k in sorted(b)])
    return sum(float(np.max(B @ a[k])) >= thresh for k in sorted(a)) / len(a)


def _texts(segs: Sequence[Segment], repr_field: str) -> list[str]:
    out = []
    for s in segs:
        v = getattr(s, repr_field, "") or ""
        out.append(v if v.strip() else s.masked)
    return out


def mine(segments: Sequence[Segment], repr_field: str = "masked", min_size: int = 20,
         min_samples: int = 10, selection: str = "eom", dims: int = 10, seed: int = 0,
         top_k: int = 8, n_exemplars: int = 3, X: np.ndarray | None = None) -> ClusterResult:
    """One clustering run. Pass X to reuse embeddings across settings."""
    texts = _texts(segments, repr_field)
    X = embed(texts) if X is None else X
    Z = reduce_dims(X, n=dims, seed=seed)
    labels = hdbscan(Z, min_size, min_samples, selection)
    docs: dict[int, list[str]] = {}
    for t, k in zip(texts, labels.tolist()):
        if k != -1:
            docs.setdefault(k, []).append(t)
    terms = ctfidf(docs, top_k=top_k) if docs else {}
    out = []
    for k in sorted(docs):
        idx = np.where(labels == k)[0]
        med = _medoids(X, idx, n_exemplars)
        out.append(Cluster(id=int(k), size=int(len(idx)), terms=terms.get(k, []),
                           exemplars=[texts[i] for i in med],
                           exemplar_raw=[segments[i].text[:600] for i in med]))
    return ClusterResult(clusters=out, labels=labels.tolist(),
                         n_noise=int((labels == -1).sum()),
                         params={"repr": repr_field, "min_cluster_size": min_size,
                                 "min_samples": min_samples, "cluster_selection": selection,
                                 "dims": dims, "seed": seed, "n_segments": len(segments)})


def sweep(segments: Sequence[Segment], sizes: Sequence[int] = (10, 20, 40, 80),
          selections: Sequence[str] = ("eom", "leaf"), repr_field: str = "masked",
          **kw) -> list[dict]:
    """What each granularity yields on your corpus. Embeds once."""
    X = embed(_texts(segments, repr_field))
    rows = []
    for sel in selections:
        for s in sizes:
            r = mine(segments, repr_field=repr_field, min_size=s, selection=sel, X=X, **kw)
            rows.append({"cluster_selection": sel, "min_cluster_size": s,
                         "n_clusters": len(r.clusters), "n_noise": r.n_noise,
                         "noise_frac": round(r.n_noise / max(len(segments), 1), 3),
                         "sizes": [c.size for c in r.clusters]})
    return rows


def stability(segments: Sequence[Segment], repr_field: str = "masked", seed: int = 0,
              thresh: float = 0.80, **kw) -> float:
    """Split-half stability: cluster each half, report the fraction of half-A clusters that have
    a centroid partner in half-B. This is the number that says whether a granularity found
    structure or noise; below ~0.8 the clusters are not reproducible."""
    X = embed(_texts(segments, repr_field))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(segments))
    half = len(segments) // 2
    a_idx, b_idx = order[:half], order[half:]
    cents = []
    for idx in (a_idx, b_idx):
        sub = [segments[i] for i in idx]
        r = mine(sub, repr_field=repr_field, X=X[idx], seed=seed, **kw)
        cents.append(centroids(X[idx], np.asarray(r.labels)))
    return round(match_centroids(cents[0], cents[1], thresh), 3)

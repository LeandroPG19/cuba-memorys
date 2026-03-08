import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFIndex:

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self._matrix: np.ndarray | None = None
        self._corpus: list[str] = []
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            self._fitted = False
            self._matrix = None
            self._corpus = []
            return

        self._corpus = corpus
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True

    def query(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self._fitted or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0
        ]

    def similarity(self, text_a: str, text_b: str) -> float:
        if self._fitted:
            vecs = self._vectorizer.transform([text_a, text_b])
        else:
            temp = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2),
                sublinear_tf=True, strip_accents="unicode",
            )
            vecs = temp.fit_transform([text_a, text_b])

        score = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
        return max(0.0, min(1.0, float(score)))

    def clear(self) -> None:
        self._matrix = None
        self._corpus = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

tfidf_index = TFIDFIndex()

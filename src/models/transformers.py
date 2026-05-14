"""Picklable sklearn transformers — ayrı modülde yaşar, hangi script'ten çalıştırıldığından
bağımsız olarak `src.models.transformers.CatBoostPrep` olarak unpickle edilebilir.
"""
from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CatBoostPrep(BaseEstimator, TransformerMixin):
    """Numerik kolonları median impute, kategorik kolonları string + 'NA' fill.

    CatBoost native categorical handling kullanır; OrdinalEncoder yerine doğrudan
    string ile besler.
    """

    def __init__(self, num_cols, cat_cols):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self._num_med = None

    def fit(self, X, y=None):
        self._num_med = X[self.num_cols].median(numeric_only=True)
        return self

    def transform(self, X):
        out = X.copy()
        out[self.num_cols] = out[self.num_cols].fillna(self._num_med)
        for c in self.cat_cols:
            out[c] = out[c].astype("string").fillna("NA")
        return out[self.num_cols + self.cat_cols]

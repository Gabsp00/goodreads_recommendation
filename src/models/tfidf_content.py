"""
Técnica 1: Recomendação baseada em conteúdo usando TF-IDF.

Fluxo esperado:

1. Ler books.csv (BOOKS_PATH) -> catálogo de livros.
2. Ajustar um vetor TF-IDF sobre o campo "text".
3. Carregar interactions_train.csv para ter o histórico dos usuários.
4. Para um usuário:
   - pegar os livros lidos no treino;
   - construir um vetor de perfil (média dos vetores TF-IDF desses livros);
   - calcular similaridade cosseno com todos os livros;
   - retornar top-k livros, excluindo os já vistos.

5. Para uma consulta em texto livre:
   - transformar o texto em vetor TF-IDF;
   - calcular similaridade cosseno com todos os livros;
   - retornar top-k livros.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    BOOKS_PATH,
    INTERACTIONS_TRAIN_PATH,
    TFIDF_NGRAM_RANGE,
    TFIDF_MAX_FEATURES,
)

@dataclass
class TFIDFConfig:
    """
    Configurações específicas do modelo TF-IDF.
    """
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE
    max_features: Optional[int] = TFIDF_MAX_FEATURES
    stop_words: Optional[str] = "english"

class TFIDFContentRecommender:
    """
    Implementação base da técnica de recomendação por conteúdo usando TF-IDF.
    """
    def __init__(self, config: Optional[TFIDFConfig] = None) -> None:
        self.config = config or TFIDFConfig()

        self.books_df: Optional[pd.DataFrame] = None
        self.interactions_train_df: Optional[pd.DataFrame] = None

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self.book_id_to_idx: dict = {}
        self.idx_to_book_id: np.ndarray | None = None

    def _load_books(self) -> pd.DataFrame:
        if self.books_df is None:
            self.books_df = pd.read_csv(BOOKS_PATH)
            self.books_df = self.books_df.reset_index(drop=True)
        return self.books_df

    def _load_interactions_train(self) -> pd.DataFrame:
        if self.interactions_train_df is None:
            self.interactions_train_df = pd.read_csv(INTERACTIONS_TRAIN_PATH)
        return self.interactions_train_df

    # Treino / construção da matriz TF-IDF
    def fit(self) -> None:
        """
        Carrega books.csv e ajusta o TF-IDF no campo 'text'.
        Também constrói os mapeamentos book_id <-> índice.
        """
        books_df = self._load_books()

        if "book_id" not in books_df.columns:
            raise ValueError("books.csv precisa ter uma coluna 'book_id'.")
        if "text" not in books_df.columns:
            raise ValueError("books.csv precisa ter uma coluna 'text' (título+descrição).")

        texts = books_df["text"].fillna("").astype(str).tolist()

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            max_features=self.config.max_features,
            stop_words=self.config.stop_words,
            norm="l2",
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        self.book_id_to_idx = {
            book_id: idx for idx, book_id in enumerate(books_df["book_id"].tolist())
        }
        self.idx_to_book_id = books_df["book_id"].values

    def _ensure_fitted(self) -> None:
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("O modelo TF-IDF ainda não foi ajustado. Chame fit() primeiro.")

    def _get_item_indices_for_user(self, user_id: str) -> List[int]:
        """
        Retorna a lista de índices (linhas da matriz TF-IDF) dos livros
        consumidos pelo usuário no conjunto de treino.
        """
        interactions = self._load_interactions_train()

        if "user_id" not in interactions.columns or "book_id" not in interactions.columns:
            raise ValueError("interactions_train.csv precisa ter colunas 'user_id' e 'book_id'.")

        user_interactions = interactions[interactions["user_id"] == user_id]
        book_ids = user_interactions["book_id"].tolist()

        indices = [
            self.book_id_to_idx[bid]
            for bid in book_ids
            if bid in self.book_id_to_idx
        ]
        return indices

    def _build_user_profile_vector(self, user_id: str):
        """
        Constrói o vetor de perfil do usuário como a média dos vetores TF-IDF
        dos livros que ele já consumiu.
        """
        self._ensure_fitted()
        assert self.tfidf_matrix is not None

        indices = self._get_item_indices_for_user(user_id)
        if not indices:
            raise ValueError(f"Usuário {user_id} não possui histórico em interactions_train.")

        user_items_matrix = self.tfidf_matrix[indices]

        user_vec = user_items_matrix.mean(axis=0)

        import numpy as np
        user_vec = np.asarray(user_vec)

        return user_vec

    def _scores_from_vector(self, query_vec) -> np.ndarray:
        """
        Calcula similaridade cosseno entre um vetor de consulta (1 x vocab)
        e todos os livros na matriz TF-IDF.
        Retorna um array 1D de scores com tamanho n_books.
        """
        self._ensure_fitted()
        assert self.tfidf_matrix is not None

        scores = cosine_similarity(query_vec, self.tfidf_matrix)
        scores = np.asarray(scores).ravel()
        return scores

    def _build_recommendation_df(
        self,
        scores: np.ndarray,
        exclude_book_ids: Optional[set] = None,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Gera um DataFrame com as top-k recomendações, dado o vetor de scores.
        """
        books_df = self._load_books()
        exclude_book_ids = exclude_book_ids or set()

        ranked_indices = np.argsort(scores)[::-1]

        filtered_indices = []
        for idx in ranked_indices:
            book_id = self.idx_to_book_id[idx]
            if book_id in exclude_book_ids:
                continue
            filtered_indices.append(idx)
            if len(filtered_indices) >= k:
                break

        recs = books_df.iloc[filtered_indices].copy()
        recs["score"] = scores[filtered_indices]
        return recs

    def recommend_for_user(
        self,
        user_id: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Gera recomendações de livros para um usuário específico.
        """
        interactions = self._load_interactions_train()
        user_items = interactions[interactions["user_id"] == user_id]["book_id"].tolist()
        user_items_set = set(user_items)

        user_vec = self._build_user_profile_vector(user_id)
        scores = self._scores_from_vector(user_vec)

        recs = self._build_recommendation_df(
            scores,
            exclude_book_ids=user_items_set,
            k=k,
        )
        return recs

    def recommend_for_query(
        self,
        query: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Gera recomendações de livros a partir de um texto de consulta livre.
        """
        self._ensure_fitted()
        assert self.vectorizer is not None

        query_vec = self.vectorizer.transform([query])
        scores = self._scores_from_vector(query_vec)

        recs = self._build_recommendation_df(scores, exclude_book_ids=set(), k=k)
        return recs
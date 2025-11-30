"""
Técnica 2: Recomendação baseada em conteúdo usando embeddings semânticos.
...
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Callable  # <-- ADICIONADO Callable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    BOOKS_PATH,
    INTERACTIONS_TRAIN_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)

@dataclass
class EmbeddingConfig:
    model_name: str = EMBEDDING_MODEL_NAME
    batch_size: int = EMBEDDING_BATCH_SIZE
    use_gpu_if_available: bool = True

class EmbeddingContentRecommender:
    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()

        self.books_df: Optional[pd.DataFrame] = None
        self.interactions_train_df: Optional[pd.DataFrame] = None

        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None  # (n_books x dim)

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

    def _init_model(self) -> None:
        if self.model is not None:
            return

        device = "cuda" if (self.config.use_gpu_if_available) else "cpu"
        self.model = SentenceTransformer(self.config.model_name, device=device)

    def fit(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Carrega books.csv, inicializa o modelo de embeddings
        e gera os vetores para o campo 'text'.

        Parameters
        ----------
        progress_callback : callable, optional
            Função chamada a cada batch de embeddings gerado.
            Assinatura: progress_callback(batches_concluidos, total_batches).
            - Se None, não reporta progresso (modo padrão, ex.: avaliação offline).
        """
        books_df = self._load_books()

        if "book_id" not in books_df.columns:
            raise ValueError("books.csv precisa ter uma coluna 'book_id'.")
        if "text" not in books_df.columns:
            raise ValueError("books.csv precisa ter uma coluna 'text'.")

        texts = books_df["text"].fillna("").astype(str).tolist()

        self._init_model()
        assert self.model is not None

        batch_size = self.config.batch_size
        n_texts = len(texts)

        if n_texts == 0:
            raise ValueError("books.csv não possui linhas para gerar embeddings.")

        batch_starts = list(range(0, n_texts, batch_size))
        total_batches = len(batch_starts)

        all_embs: List[np.ndarray] = []

        for batch_idx, start in enumerate(batch_starts):
            end = min(start + batch_size, n_texts)
            batch_texts = texts[start:end]

            batch_embs = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embs.append(batch_embs)

            if progress_callback is not None:
                progress_callback(batch_idx + 1, total_batches)

        self.embeddings = np.vstack(all_embs)

        self.book_id_to_idx = {
            book_id: idx for idx, book_id in enumerate(books_df["book_id"].tolist())
        }
        self.idx_to_book_id = books_df["book_id"].values

    def _ensure_fitted(self) -> None:
        if self.embeddings is None or self.idx_to_book_id is None:
            raise RuntimeError("O modelo de embeddings ainda não foi ajustado. Chame fit() primeiro.")

    def _get_item_indices_for_user(self, user_id: str) -> List[int]:
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

    def _build_user_embedding(self, user_id: str) -> np.ndarray:
        """
        Constrói o embedding de perfil do usuário como a média
        dos embeddings dos livros que ele já consumiu.
        """
        self._ensure_fitted()
        assert self.embeddings is not None

        indices = self._get_item_indices_for_user(user_id)
        if not indices:
            raise ValueError(f"Usuário {user_id} não possui histórico em interactions_train.")

        user_item_embs = self.embeddings[indices]
        user_vec = user_item_embs.mean(axis=0)

        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm

        return user_vec

    def _compute_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Calcula similaridade (produto interno) entre um vetor de consulta
        normalizado (dim,) e todos os livros, cujos embeddings já estão
        normalizados (embeddings).
        """
        self._ensure_fitted()
        assert self.embeddings is not None

        scores = self.embeddings @ query_vec
        return scores

    def _build_recommendation_df(
        self,
        scores: np.ndarray,
        k: int = 10,
        exclude_book_ids: Optional[set] = None,
    ) -> pd.DataFrame:
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

    def recommend_for_user(self, user_id: str, k: int = 10) -> pd.DataFrame:
        """
        Gera recomendações para um usuário, usando o histórico dele e
        a média de embeddings dos livros consumidos.
        """
        interactions = self._load_interactions_train()
        user_items = interactions[interactions["user_id"] == user_id]["book_id"].tolist()
        user_items_set = set(user_items)

        user_vec = self._build_user_embedding(user_id)
        scores = self._compute_scores(user_vec)

        recs = self._build_recommendation_df(
            scores,
            k=k,
            exclude_book_ids=user_items_set,
        )
        return recs

    def recommend_for_query(self, query: str, k: int = 10) -> pd.DataFrame:
        """
        Gera recomendações a partir de um texto de consulta livre.
        """
        self._ensure_fitted()
        self._init_model()
        assert self.model is not None

        query_vec = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]  # (dim,)

        scores = self._compute_scores(query_vec)
        recs = self._build_recommendation_df(scores, k=k, exclude_book_ids=set())
        return recs
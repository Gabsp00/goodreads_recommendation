"""
Carregamento e pré-processamento básico do UCSD Goodreads.

Responsabilidades deste módulo:
- Ler o arquivo de interações goodreads_interactions.csv
- Filtrar interações "úteis" (usuário realmente leu / avaliou)
- Aplicar filtros de:
    - MIN_USER_INTERACTIONS
    - MIN_ITEM_INTERACTIONS
    - TARGET_NUM_USERS
    - TARGET_NUM_ITEMS
- Montar a tabela de usuários (users.csv)
- Montar a tabela de livros com metadados (books.csv)
  usando:
    - book_id_map.csv (mapa id interno -> id Goodreads real)
    - goodreads_books.json.gz (título + descrição)

Este módulo NÃO faz o split train/val/test. Isso é feito em make_splits.py.
"""

from __future__ import annotations
from collections import Counter
import csv
import gzip
import json
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_DIR,
    MIN_USER_INTERACTIONS,
    MIN_ITEM_INTERACTIONS,
    MIN_RATING_FOR_POSITIVE,
    TARGET_NUM_USERS,
    TARGET_NUM_ITEMS,
    RANDOM_SEED,
)

# Nomes esperados dos arquivos brutos do UCSD Goodreads
INTERACTIONS_FILENAME = "goodreads_interactions.csv"
BOOKS_JSON_FILENAME = "goodreads_books.json.gz"
BOOK_ID_MAP_FILENAME = "book_id_map.csv"

# 1) Carregar interações brutas
def load_interactions_raw() -> pd.DataFrame:
    """
    Lê goodreads_interactions.csv e retorna um DataFrame com:
        user_id, book_id, original_idx

    - Apenas mantém interações em que o usuário LEU o livro
      (is_read == 1) ou deu rating > 0.
    - original_idx é um índice global que preserva a "ordem"
      das interações para o split temporal depois.
    """
    path = RAW_DATA_DIR / "goodreads" / INTERACTIONS_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de interações não encontrado em {path}.\n"
            "Verifique se você baixou goodreads_interactions.csv "
            "para data/raw/goodreads/."
        )

    print(f"[load_goodreads] Lendo interações de: {path}")
    interactions = pd.read_csv(path)

    expected_cols = {"user_id", "book_id", "is_read", "rating"}
    missing = expected_cols - set(interactions.columns)
    if missing:
        raise ValueError(
            f"goodreads_interactions.csv está faltando colunas obrigatórias: {missing}"
        )

    # Mantém interações que indicam algum tipo de feedback
    mask = (interactions["is_read"] == 1) | (interactions["rating"] >= MIN_RATING_FOR_POSITIVE)
    interactions = interactions.loc[mask, ["user_id", "book_id"]].copy()

    interactions["user_id"] = interactions["user_id"].astype(np.int64)
    interactions["book_id"] = interactions["book_id"].astype(np.int64)

    # índice global para preservar uma "ordem"
    interactions["original_idx"] = np.arange(len(interactions), dtype=np.int64)

    print(
        f"[load_goodreads] Interações após filtro is_read/rating: {len(interactions):,} "
        f"(users={interactions['user_id'].nunique():,}, "
        f"books={interactions['book_id'].nunique():,})"
    )
    return interactions

# 2) Filtrar usuários/itens por frequência e reduzir para 40k x 30k
def filter_users_and_items(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica:
      - filtro iterativo de MIN_USER_INTERACTIONS / MIN_ITEM_INTERACTIONS
      - redução para TARGET_NUM_USERS / TARGET_NUM_ITEMS
        (mantendo os mais frequentes).

    Retorna um novo DataFrame filtrado, ainda com a coluna original_idx.
    """
    df = interactions.copy()

    print("[load_goodreads] Aplicando filtro iterativo de usuários/itens...")

    while True:
        n_before = len(df)
        # Filtra usuários com poucas interações
        user_counts = df["user_id"].value_counts()
        good_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        df = df[df["user_id"].isin(good_users)]

        # Filtra itens com poucas interações
        item_counts = df["book_id"].value_counts()
        good_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        df = df[df["book_id"].isin(good_items)]

        n_after = len(df)
        print(
            f"[load_goodreads]  Iteração filtro freq: {n_before:,} -> {n_after:,} interações"
        )
        if n_after == n_before:
            break

    print(
        f"[load_goodreads] Após filtros mínimos: "
        f"{df['user_id'].nunique():,} usuários, "
        f"{df['book_id'].nunique():,} livros, "
        f"{len(df):,} interações"
    )

    # Reduz para TARGET_NUM_USERS
    user_counts = df["user_id"].value_counts()
    if len(user_counts) > TARGET_NUM_USERS:
        top_users = user_counts.sort_values(ascending=False).head(TARGET_NUM_USERS).index
        df = df[df["user_id"].isin(top_users)]
        print(
            f"[load_goodreads] Após limitar para {TARGET_NUM_USERS} usuários: "
            f"{df['user_id'].nunique():,} usuários, {len(df):,} interações"
        )

    # Reduz para TARGET_NUM_ITEMS
    item_counts = df["book_id"].value_counts()
    if len(item_counts) > TARGET_NUM_ITEMS:
        top_items = item_counts.sort_values(ascending=False).head(TARGET_NUM_ITEMS).index
        df = df[df["book_id"].isin(top_items)]
        print(
            f"[load_goodreads] Após limitar para {TARGET_NUM_ITEMS} livros: "
            f"{df['book_id'].nunique():,} livros, {len(df):,} interações"
        )

    # Reordena só pra ficar organizado
    df = df.reset_index(drop=True)

    return df

# 3) Construir tabela de usuários
def build_users_table(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Cria um DataFrame users_df com uma coluna:
        - user_id
    a partir das interações filtradas.
    """
    users = (
        interactions["user_id"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    users_df = pd.DataFrame({"user_id": users})
    print(f"[load_goodreads] Tabela de usuários: {len(users_df):,} linhas")
    return users_df

# 4) Construir metadados de livros (books_df) usando JSON + book_id_map.csv
def _load_book_id_map() -> pd.DataFrame:
    """
    Lê book_id_map.csv e retorna um DataFrame com colunas:
        - book_id_internal  (ID inteiro usado em goodreads_interactions.csv)
        - book_id_external  (ID real do Goodreads, usado em goodreads_books.json.gz)
    """
    path = RAW_DATA_DIR / "goodreads" / BOOK_ID_MAP_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"book_id_map.csv não encontrado em {path}.\n"
            "Baixe o arquivo para data/raw/goodreads/."
        )

    print(f"[load_goodreads] Lendo mapa de IDs de livros de: {path}")
    m = pd.read_csv(path)

    if m.shape[1] < 2:
        raise ValueError("book_id_map.csv precisa ter pelo menos 2 colunas.")

    internal_col, external_col = m.columns[:2]

    m = m[[internal_col, external_col]].copy()
    m = m.rename(
        columns={
            internal_col: "book_id_internal",
            external_col: "book_id_external",
        }
    )

    m["book_id_internal"] = m["book_id_internal"].astype(np.int64)
    m["book_id_external"] = m["book_id_external"].astype(np.int64)

    return m

def build_books_metadata(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    A partir das interações filtradas:
      - pega o conjunto de book_id internos
      - usa book_id_map.csv para obter os IDs externos equivalentes
      - lê goodreads_books.json.gz linha a linha e guarda apenas
        os livros que estão em uso
      - devolve um DataFrame com colunas: book_id, title, description
        onde book_id é o ID INTERNO (batendo com interactions.book_id).
    """
    internal_ids = set(interactions["book_id"].unique())
    print(
        f"[load_goodreads] Construindo metadados de livros para "
        f"{len(internal_ids):,} IDs internos..."
    )

    book_id_map = _load_book_id_map()
    book_id_map = book_id_map[book_id_map["book_id_internal"].isin(internal_ids)].copy()

    wanted_external_ids = set(book_id_map["book_id_external"].tolist())
    print(
        f"[load_goodreads] IDs externos de livros necessários: "
        f"{len(wanted_external_ids):,}"
    )

    books_json_path = RAW_DATA_DIR / "goodreads" / BOOKS_JSON_FILENAME
    if not books_json_path.exists():
        raise FileNotFoundError(
            f"Arquivo de livros não encontrado em {books_json_path}.\n"
            "Baixe goodreads_books.json.gz para data/raw/goodreads/."
        )

    print(f"[load_goodreads] Lendo metadados de livros de: {books_json_path}")
    rows = []

    with gzip.open(books_json_path, "rt", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            ext_id = int(book["book_id"])
            if ext_id in wanted_external_ids:
                rows.append(
                    {
                        "book_id_external": ext_id,
                        "title": book.get("title", ""),
                        "description": book.get("description", ""),
                    }
                )

    if not rows:
        raise RuntimeError(
            "Nenhum livro do conjunto filtrado foi encontrado em goodreads_books.json.gz. "
            "Verifique se book_id_map.csv e goodreads_books.json.gz "
            "são compatíveis com o goodreads_interactions.csv que você baixou."
        )

    books_meta = pd.DataFrame(rows)
    books_merged = book_id_map.merge(books_meta, on="book_id_external", how="inner")

    books_final = (
        books_merged[["book_id_internal", "title", "description"]]
        .drop_duplicates(subset=["book_id_internal"])
        .rename(columns={"book_id_internal": "book_id"})
        .reset_index(drop=True)
    )

    print(
        f"[load_goodreads] Tabela de livros final: {len(books_final):,} linhas "
        f"(metadata encontrada para "
        f"{len(books_final['book_id'].unique()):,} livros)."
    )

    missing = internal_ids - set(books_final["book_id"].unique())
    if missing:
        print(
            f"[load_goodreads] Aviso: {len(missing):,} livros presentes em interactions "
            "não apareceram em goodreads_books.json.gz (sem metadata)."
        )

    return books_final
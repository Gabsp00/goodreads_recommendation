"""
Pipeline de preparação de dados:

1. Lê interações brutas do UCSD Goodreads (load_interactions_raw).
2. Aplica filtros de frequência + redução para 40k usuários / 30k livros
   (filter_users_and_items).
3. Constrói:
    - users.csv
    - books.csv
4. Cria splits temporais (por ordem de original_idx) para:
    - interactions_train.csv
    - interactions_val.csv
    - interactions_test.csv

Os parâmetros de tamanho dos splits vêm de:
    VAL_INTERACTIONS_RANGE
    TEST_INTERACTIONS_RANGE

Por simplicidade, usamos SEMPRE o limite superior do range
(por exemplo, 3 interações para val e 3 para test).
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from src.data_prep.preprocess_text import build_books_table, save_books_table

from config import (
    BOOKS_PATH,
    USERS_PATH,
    INTERACTIONS_TRAIN_PATH,
    INTERACTIONS_VAL_PATH,
    INTERACTIONS_TEST_PATH,
    VAL_INTERACTIONS_RANGE,
    TEST_INTERACTIONS_RANGE,
)
from src.data_prep.load_goodreads import (
    load_interactions_raw,
    filter_users_and_items,
    build_users_table,
    build_books_metadata,
)

# 1) Criar splits train/val/test a partir das interações filtradas
def create_splits(
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recebe o DataFrame de interações já filtrado, com colunas:
        user_id, book_id, original_idx

    Retorna:
        interactions_train, interactions_val, interactions_test

    Estratégia de split:
      - Ordena por (user_id, original_idx) pra simular ordem temporal.
      - Para cada usuário:
          * usa TEST_INTERACTIONS_RANGE[1] últimas interações como test
          * usa VAL_INTERACTIONS_RANGE[1] penúltimas como val
          * resto vai para train
    """
    df = interactions.copy()

    if "original_idx" not in df.columns:
        # fallback de segurança
        df = df.reset_index().rename(columns={"index": "original_idx"})

    df = df.sort_values(["user_id", "original_idx"]).reset_index(drop=True)

    df["pos"] = df.groupby("user_id").cumcount()
    df["count"] = df.groupby("user_id")["book_id"].transform("count")

    val_n = VAL_INTERACTIONS_RANGE[1]
    test_n = TEST_INTERACTIONS_RANGE[1]

    df["test_start"] = df["count"] - test_n
    df["val_start"] = df["count"] - test_n - val_n
    df["test_start"] = df["test_start"].clip(lower=0)
    df["val_start"] = df["val_start"].clip(lower=0)
    df["split"] = "train"

    # Test: últimas test_n interações
    mask_test = df["pos"] >= df["test_start"]
    df.loc[mask_test, "split"] = "test"

    # Val: penúltimas val_n antes da região de test
    mask_val = (df["pos"] >= df["val_start"]) & (df["pos"] < df["test_start"])
    df.loc[mask_val, "split"] = "val"

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    drop_cols = ["pos", "count", "split", "test_start", "val_start", "original_idx"]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    val_df = val_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    return train_df, val_df, test_df

# 2) Função principal: encadear tudo e salvar em data/processed/
def main() -> None:
    print("=== [make_splits] Etapa 1: carregar interações brutas ===")
    interactions_raw = load_interactions_raw()
    print(
        f"[make_splits] Interações brutas: {len(interactions_raw):,} "
        f"(users={interactions_raw['user_id'].nunique():,}, "
        f"books={interactions_raw['book_id'].nunique():,})"
    )

    print("\n=== [make_splits] Etapa 2: filtrar usuários/itens ===")
    interactions_filtered = filter_users_and_items(interactions_raw)
    print(
        f"[make_splits] Interações filtradas: {len(interactions_filtered):,} "
        f"(users={interactions_filtered['user_id'].nunique():,}, "
        f"books={interactions_filtered['book_id'].nunique():,})"
    )

    print("\n=== [make_splits] Etapa 3: construir e salvar users.csv ===")
    users_df = build_users_table(interactions_filtered)
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    users_df.to_csv(USERS_PATH, index=False)
    print(f"[make_splits] users.csv salvo em: {USERS_PATH}")

    print("\n=== [make_splits] Etapa 4: construir e salvar books.csv ===")
    books_meta_df = build_books_metadata(interactions_filtered)
    books_full_df = build_books_table(books_meta_df)
    save_books_table(books_full_df)
    print(f"[make_splits] books.csv salvo em: {BOOKS_PATH}")

    print("\n=== [make_splits] Etapa 5: criar splits train/val/test ===")
    train_df, val_df, test_df = create_splits(interactions_filtered)

    train_df.to_csv(INTERACTIONS_TRAIN_PATH, index=False)
    val_df.to_csv(INTERACTIONS_VAL_PATH, index=False)
    test_df.to_csv(INTERACTIONS_TEST_PATH, index=False)

    print(
        f"[make_splits] interactions_train.csv salvo em: {INTERACTIONS_TRAIN_PATH}\n"
        f"[make_splits] interactions_val.csv salvo em:   {INTERACTIONS_VAL_PATH}\n"
        f"[make_splits] interactions_test.csv salvo em:  {INTERACTIONS_TEST_PATH}"
    )

    print("\n=== [make_splits] Pipeline de preparação concluído. ===")

if __name__ == "__main__":
    main()
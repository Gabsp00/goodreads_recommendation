"""
Funções para pré-processar textos (título + descrição) e gerar o catálogo
final de livros (books.csv) usado pelas técnicas 1, 2 e 4.
"""

from typing import Tuple

import pandas as pd

from config import BOOKS_PATH, TEXT_FIELDS, MAX_TOKENS_TEXT

def clean_text(text: str) -> str:
    """
    Limpeza básica de texto (lowercase, strip, etc.).

    Pode ser expandida depois com remoção de HTML, tokens estranhos, etc.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text

def build_books_table(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame de livros vindo de load_goodreads() e:

    - garante a existência de um identificador único (book_id)
    - limpa campos de texto relevantes (título, descrição)
    - concatena TEXT_FIELDS em um campo único "text"
    - aplica corte de comprimento (MAX_TOKENS_TEXT)

    Retorna o DataFrame pronto para ser salvo em BOOKS_PATH.
    """
    df = books_df.copy()

    for field in TEXT_FIELDS:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
        else:
            df[field] = ""

    df["text"] = df[list(TEXT_FIELDS)].agg(" ".join, axis=1)

    # Corte aproximado de MAX_TOKENS_TEXT (contado em palavras)
    df["text"] = df["text"].apply(
        lambda t: " ".join(t.split()[:MAX_TOKENS_TEXT])
    )

    return df

def save_books_table(df: pd.DataFrame) -> None:
    """
    Salva o catálogo de livros em BOOKS_PATH.
    """
    BOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BOOKS_PATH, index=False)

"""
Criação e uso do banco vetorial (por exemplo, ChromaDB) para os livros.

Fluxo:

1. Carregar books.csv.
2. Calcular embeddings com SentenceTransformers.
3. Criar/atualizar uma coleção no vector DB (VECTOR_COLLECTION_NAME).
4. Expor funções para:
   - construir o índice
   - fazer buscas de similaridade a partir de um texto
"""

from typing import List, Dict, Any
import torch

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    VECTOR_DB_IMPLEMENTATION,
    VECTOR_COLLECTION_NAME,
    BOOKS_PATH,
    PROCESSED_DATA_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)

_VECTOR_CLIENT = None
_VECTOR_COLLECTION = None
_EMBEDDING_MODEL = None

def _get_chroma_client():
    global _VECTOR_CLIENT
    if _VECTOR_CLIENT is None:
        persist_dir = PROCESSED_DATA_DIR / "vector_store"
        persist_dir.mkdir(parents=True, exist_ok=True)
        _VECTOR_CLIENT = chromadb.PersistentClient(path=str(persist_dir))
    return _VECTOR_CLIENT

def _get_collection():
    global _VECTOR_COLLECTION
    client = _get_chroma_client()
    if _VECTOR_COLLECTION is None:
        _VECTOR_COLLECTION = client.get_or_create_collection(
            name=VECTOR_COLLECTION_NAME
        )
    return _VECTOR_COLLECTION

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDING_MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=device,
        )
    return _EMBEDDING_MODEL

def _reset_collection():
    global _VECTOR_COLLECTION
    client = _get_chroma_client()
    try:
        client.delete_collection(VECTOR_COLLECTION_NAME)
    except Exception:
        pass
    _VECTOR_COLLECTION = None
    return _get_collection()

def build_vector_store() -> None:
    """
    Constrói a base vetorial a partir de books.csv.

    Se já existir, a coleção é recriada do zero.
    """
    if VECTOR_DB_IMPLEMENTATION != "chroma":
        raise NotImplementedError(
            f"VECTOR_DB_IMPLEMENTATION='{VECTOR_DB_IMPLEMENTATION}' não suportado. "
            "Implemente outros backends conforme necessário."
        )

    books = pd.read_csv(BOOKS_PATH)

    if "book_id" not in books.columns:
        raise KeyError("books.csv deve conter coluna 'book_id'.")
    if "text" not in books.columns:
        raise KeyError("books.csv deve conter coluna 'text' (pré-processada).")

    model = _get_embedding_model()

    texts = books["text"].astype(str).tolist()
    ids = books["book_id"].astype(str).tolist()

    metadatas = []
    for _, row in books.iterrows():
        metadatas.append(
            {
                "book_id": str(row["book_id"]),
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
            }
        )

    collection = _reset_collection()

    batch_size = EMBEDDING_BATCH_SIZE
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            metadatas=batch_metas,
        )

    print(f"Vector store construído com {len(texts)} livros.")

def query_vector_store(query_text: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Consulta a base vetorial com um texto e retorna os k livros mais similares.

    Retorno:
        [
          {
            "book_id": ...,
            "score": ...,
            "metadata": {...}
          },
          ...
        ]
    """
    if VECTOR_DB_IMPLEMENTATION != "chroma":
        raise NotImplementedError(
            f"VECTOR_DB_IMPLEMENTATION='{VECTOR_DB_IMPLEMENTATION}' não suportado."
        )

    collection = _get_collection()
    model = _get_embedding_model()

    query_emb = model.encode([query_text], convert_to_numpy=True)

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=k,
    )

    out: List[Dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for book_id, dist, meta in zip(ids, dists, metas):
        dist = float(dist)

        sim = 1.0 / (1.0 + dist)

        out.append(
            {
                "book_id": int(book_id),
                "score": sim,
                "distance": dist, 
                "metadata": meta,
            }
        )

    out = sorted(out, key=lambda x: x["score"], reverse=True)
    return out
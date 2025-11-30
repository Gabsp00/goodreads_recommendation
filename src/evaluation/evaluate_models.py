"""
Módulo de avaliação dos modelos de recomendação.

Responsabilidades:

- Carregar interações de teste.
- Para cada técnica (TF-IDF, Embeddings, etc.):
    - gerar recomendações top-k por usuário
    - calcular Precision@k, Recall@k, NDCG@k
    - salvar resultados em /results/{nome}_metrics.json

Implementa:
- evaluate_tfidf_model()
- evaluate_embedding_model()
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable
import traceback

import json
import pandas as pd

from config import (
    INTERACTIONS_TRAIN_PATH,
    INTERACTIONS_TEST_PATH,
    BOOKS_PATH,
    RESULTS_DIR,
    TOP_K_VALUES,
)
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.models.tfidf_content import TFIDFContentRecommender
from src.models.embedding_content import EmbeddingContentRecommender

def _load_test_interactions() -> pd.DataFrame:
    """
    Carrega o arquivo de interações de teste.
    Espera colunas pelo menos: user_id, book_id.
    """
    df = pd.read_csv(INTERACTIONS_TEST_PATH)
    required_cols = {"user_id", "book_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"interactions_test.csv está faltando colunas: {missing}"
        )
    return df

def _evaluate_generic_model(
    get_recs_for_user,
    test_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """
    Função genérica de avaliação.

    Parâmetros:
    - get_recs_for_user: função user_id -> lista de book_ids recomendados
                         (ou DataFrame com coluna 'book_id')
                         de tamanho >= max(TOP_K_VALUES)
    - test_df: DataFrame de interações de teste (user_id, book_id, ...)
    - model_name: nome da técnica (ex.: "tfidf_content")

    Retorna:
    - dicionário com métricas médias por usuário.
    """
    user_ids = test_df["user_id"].unique()
    max_k = max(TOP_K_VALUES)

    # Acumuladores
    metric_sums: Dict[str, float] = {}
    for k in TOP_K_VALUES:
        metric_sums[f"precision@{k}"] = 0.0
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"ndcg@{k}"] = 0.0

    num_users_used = 0

    for user_id in user_ids:
        user_test_items = test_df[test_df["user_id"] == user_id]["book_id"].tolist()
        relevant_items = set(user_test_items)
        if not relevant_items:
            continue

        try:
            recs = get_recs_for_user(user_id, k=max_k)
        except ValueError:
            continue

        if isinstance(recs, pd.DataFrame):
            if "book_id" not in recs.columns:
                raise ValueError(
                    f"O DataFrame de recomendações do modelo {model_name} "
                    "não contém coluna 'book_id'."
                )
            recommended_items = recs["book_id"].tolist()
        else:
            recommended_items = list(recs)

        if not recommended_items:
            continue

        for k in TOP_K_VALUES:
            rec_k = recommended_items[:k]
            p = precision_at_k(rec_k, relevant_items, k)
            r = recall_at_k(rec_k, relevant_items, k)
            n = ndcg_at_k(rec_k, relevant_items, k)

            metric_sums[f"precision@{k}"] += p
            metric_sums[f"recall@{k}"] += r
            metric_sums[f"ndcg@{k}"] += n

        num_users_used += 1

    if num_users_used == 0:
        raise RuntimeError(
            f"Nenhum usuário foi avaliado para o modelo {model_name}. "
            "Verifique se há interações de teste e histórico em train."
        )

    metrics_avg: Dict[str, Any] = {
        "model": model_name,
        "num_users": int(num_users_used),
    }
    for name, total in metric_sums.items():
        metrics_avg[name] = total / num_users_used

    return metrics_avg

def evaluate_tfidf_model() -> Dict[str, Any]:
    """
    Avalia a Técnica 1 (TF-IDF content-based) no conjunto de teste.
    """
    print("Carregando interações de teste...")
    test_df = _load_test_interactions()

    print("Carregando e ajustando modelo TF-IDF...")
    tfidf_rec = TFIDFContentRecommender()
    tfidf_rec.fit()

    def get_recs_for_user(user_id, k):
        recs_df = tfidf_rec.recommend_for_user(user_id, k=k)
        return recs_df

    print("Avaliando modelo TF-IDF...")
    metrics = _evaluate_generic_model(
        get_recs_for_user,
        test_df,
        model_name="tfidf_content",
    )
    return metrics

def evaluate_embedding_model() -> Dict[str, Any]:
    """
    Avalia a Técnica 2 (embeddings semânticos) no conjunto de teste.
    """
    print("Carregando interações de teste...")
    test_df = _load_test_interactions()

    print("Carregando e ajustando modelo de embeddings...")
    emb_rec = EmbeddingContentRecommender()
    emb_rec.fit()

    def get_recs_for_user(user_id, k):
        recs_df = emb_rec.recommend_for_user(user_id, k=k)
        return recs_df

    print("Avaliando modelo de embeddings...")
    metrics = _evaluate_generic_model(
        get_recs_for_user,
        test_df,
        model_name="embedding_content",
    )
    return metrics

# Técnica 3 – Neural CF
try:
    from src.models.neural_cf import generate_recommendations_for_all_users
except ImportError:
    generate_recommendations_for_all_users = None

# Técnica 4 – RAG (retriever)
try:
    from src.rag.vector_store import query_vector_store
except ImportError:
    query_vector_store = None

def _build_relevant_items_by_user(test_df: pd.DataFrame) -> Dict[Any, set[int]]:
    """user_id -> conjunto de book_ids relevantes a partir do conjunto de teste."""
    relevant_by_user: Dict[Any, set[int]] = {}
    for user_id, group in test_df.groupby("user_id"):
        items = set(group["book_id"].astype(int).tolist())
        relevant_by_user[user_id] = items
    return relevant_by_user

def _compute_global_metrics(
    recs_by_user: Dict[Any, Iterable[int]],
    relevant_by_user: Dict[Any, set[int]],
) -> Dict[str, Dict[int, float]]:
    """
    Calcula precision/recall/NDCG médios para TOP_K_VALUES.

    Retorna:

    {
      "precision": {5: 0.12, 10: 0.10},
      "recall": {5: 0.08, 10: 0.15},
      "ndcg": {5: 0.10, 10: 0.12},
    }
    """
    recs_by_user_norm: Dict[Any, list[int]] = {
        u: list(map(int, items)) for u, items in recs_by_user.items()
    }

    metrics: Dict[str, Dict[int, float]] = {
        "precision": {},
        "recall": {},
        "ndcg": {},
    }

    users = sorted(
        set(recs_by_user_norm.keys()).intersection(relevant_by_user.keys())
    )
    num_users = len(users)
    if num_users == 0:
        for k in TOP_K_VALUES:
            metrics["precision"][k] = 0.0
            metrics["recall"][k] = 0.0
            metrics["ndcg"][k] = 0.0
        return metrics

    for k in TOP_K_VALUES:
        prec_vals = []
        rec_vals = []
        ndcg_vals = []

        for u in users:
            recs = recs_by_user_norm.get(u, [])
            relevant = relevant_by_user.get(u, set())
            prec_vals.append(precision_at_k(recs, relevant, k))
            rec_vals.append(recall_at_k(recs, relevant, k))
            ndcg_vals.append(ndcg_at_k(recs, relevant, k))

        metrics["precision"][k] = float(sum(prec_vals) / len(prec_vals))
        metrics["recall"][k] = float(sum(rec_vals) / len(rec_vals))
        metrics["ndcg"][k] = float(sum(ndcg_vals) / len(ndcg_vals))

    return metrics

# Avaliação Técnica 3 – Neural CF
def evaluate_neural_cf() -> Dict[str, Any]:
    """
    Avalia o modelo de Collaborative Filtering neural.

    Pressupõe que:

    - Já existe um modelo treinado e salvo em disco; e
    - src/models/neural_cf.py expõe:

        generate_recommendations_for_all_users(k: int)
            -> {user_id: [book_id, ...]}
    """
    print("*** avaliando neural_cf...***")
    if generate_recommendations_for_all_users is None:
        raise RuntimeError(
            "Função generate_recommendations_for_all_users não disponível. "
            "Verifique src/models/neural_cf.py."
        )

    test_df = pd.read_csv(INTERACTIONS_TEST_PATH)
    print(test_df.head())
    print(test_df.columns)
    relevant_by_user = _build_relevant_items_by_user(test_df)

    max_k = max(TOP_K_VALUES)
    print(f"gerando recomendações: k={max_k}")

    recs_by_user = generate_recommendations_for_all_users(k=max_k)
    print("TIPO de uma chave de recs_by_user:",
      type(next(iter(recs_by_user.keys()))))
    print("Exemplo de chave de recs_by_user:",
      list(recs_by_user.keys())[:5])

    print("TIPO de uma chave de relevant_by_user:",
      type(next(iter(relevant_by_user.keys()))))
    print("Exemplo de chave de relevant_by_user:",
      list(relevant_by_user.keys())[:5])
    print(f"computando métricas ...")

    metrics = _compute_global_metrics(recs_by_user, relevant_by_user)

    result: Dict[str, Any] = {
        "model": "neural_cf",
        "num_users": len(relevant_by_user),
        "top_k": list(TOP_K_VALUES),
        "metrics": metrics,
    }
    return result

# Avaliação Técnica 4 – RAG (retriever por usuário)
def _build_user_profiles_from_train(
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    max_books_per_user: int = 10,
) -> Dict[Any, str]:
    """
    Cria um texto de perfil por usuário a partir dos livros do treino.

    Estratégia simples:
    - Para cada usuário, pega até `max_books_per_user` livros do treino.
    - Concatena título + descrição de cada livro em um texto único.
    """
    books = books_df.set_index("book_id")

    user_profiles: Dict[Any, str] = {}
    for user_id, group in train_df.groupby("user_id"):
        book_ids = group["book_id"].astype(int).tolist()[:max_books_per_user]
        texts: list[str] = []

        for bid in book_ids:
            if bid not in books.index:
                continue

            row = books.loc[bid]

            raw_text = row.get("text", "")
            if pd.isna(raw_text):
                raw_text = ""
            raw_text = str(raw_text).strip()

            if raw_text:
                piece = raw_text
            else:
                title = str(row.get("title", "")).strip()
                desc = str(row.get("description", "")).strip()
                piece = f"{title}. {desc}".strip()

            if piece:
                texts.append(piece)

        if texts:
            user_profiles[user_id] = " ".join(texts)

    return user_profiles

def evaluate_rag_retriever() -> Dict[str, Any]:
    """
    Avalia a parte de recuperação vetorial usada na Técnica 4 (RAG).

    Aqui não chamamos o LLM – só medimos a qualidade do *retriever*:

    - Construímos um "perfil textual" para cada usuário a partir dos livros
      do conjunto de treino.
    - Usamos esse texto como query no vector_store (`query_vector_store`).
    - Pegamos os top-k `book_id` para cada usuário.
    - Comparamos com os livros do conjunto de teste desse usuário.
    """
    print("*** avaliando rag_retriever ...***")

    if query_vector_store is None:
        raise RuntimeError(
            "Função query_vector_store não disponível. "
            "Verifique src/rag/vector_store.py."
        )

    train_df = pd.read_csv(INTERACTIONS_TRAIN_PATH)
    test_df = pd.read_csv(INTERACTIONS_TEST_PATH)
    books_df = pd.read_csv(BOOKS_PATH)

    relevant_by_user = _build_relevant_items_by_user(test_df)
    user_profiles = _build_user_profiles_from_train(train_df, books_df)

    users_to_eval = sorted(
        set(user_profiles.keys()).intersection(relevant_by_user.keys())
    )

    max_k = max(TOP_K_VALUES)
    recs_by_user: Dict[Any, list[int]] = {}

    # Itens já vistos no treino (para não recomendar o que a pessoa já leu)
    train_items_by_user: Dict[Any, set[int]] = (
        train_df.groupby("user_id")["book_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )
    print("Obtendo recomendações por usuário ")

    for user_id in users_to_eval:
        query_text = user_profiles[user_id]

        retrieved = query_vector_store(query_text, k=max_k * 2)
        all_book_ids = [int(r["book_id"]) for r in retrieved]

        seen = train_items_by_user.get(user_id, set())
        filtered = [bid for bid in all_book_ids if bid not in seen]

        recs_by_user[user_id] = filtered[:max_k]

    metrics = _compute_global_metrics(recs_by_user, relevant_by_user)

    result: Dict[str, Any] = {
        "model": "rag_retriever",
        "num_users": len(users_to_eval),
        "top_k": list(TOP_K_VALUES),
        "metrics": metrics,
    }
    return result

def save_metrics(metrics: Dict[str, Any], name: str) -> None:
    """
    Salva dicionário de métricas em results/{name}_metrics.json.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{name}_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {output_path}")

if __name__ == "__main__":
    try:
        print("=== Avaliando TF-IDF ===")
        tfidf_metrics = evaluate_tfidf_model()
        save_metrics(tfidf_metrics, name="tfidf")

        print("\n=== Avaliando Embeddings ===")
        emb_metrics = evaluate_embedding_model()
        save_metrics(emb_metrics, name="embedding")

        print("\nAvaliação concluída.")

    except Exception as e:
        print("Erro durante a avaliação:", repr(e))

    try:
        neural_metrics = evaluate_neural_cf()
        save_metrics(neural_metrics, name="neural_cf")
        print(
            "Métricas da Técnica 3 (Neural CF) salvas em "
            "results/neural_cf_metrics.json"
        )
    except Exception as e:
        print(f"Falha ao avaliar Neural CF: {e}")
        traceback.print_exc()

    try:
        rag_metrics = evaluate_rag_retriever()
        save_metrics(rag_metrics, name="rag_retriever")
        print(
            "Métricas da Técnica 4 (RAG - retriever) salvas em "
            "results/rag_retriever_metrics.json"
        )
    except Exception as e:
        print(f"Falha ao avaliar RAG retriever: {e}")
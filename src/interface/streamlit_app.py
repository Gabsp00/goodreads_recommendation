from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Any
import os 
import pandas as pd
import streamlit as st
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    INTERACTIONS_TRAIN_PATH,
    INTERACTIONS_VAL_PATH,
    INTERACTIONS_TEST_PATH,
    BOOKS_PATH,
    USERS_PATH,
    RAW_DATA_DIR,
    RESULTS_DIR,
    GEMINI_API_KEY_ENVVAR,
)
from src.models.tfidf_content import TFIDFContentRecommender
from src.models.embedding_content import EmbeddingContentRecommender
from src.evaluation.evaluate_models import evaluate_rag_retriever, save_metrics
from src.rag.rag_recommender import recommend_with_rag_for_query
from src.rag.llm_client import generate_llm_response
from src.models.neural_cf import recommend_for_user_neural_cf

st.set_page_config(
    page_title="Recomendador Goodreads - Projeto AM",
    layout="wide",
)

TFIDF_METRICS_PATH = RESULTS_DIR / "tfidf_metrics.json"
EMB_METRICS_PATH = RESULTS_DIR / "embedding_metrics.json"
NEURAL_CF_METRICS_PATH = RESULTS_DIR / "neural_cf_metrics.json"
GOODREADS_RAW_DIR = RAW_DATA_DIR / "goodreads"
RAG_METRICS_FILE = RESULTS_DIR / "rag_retriever_metrics.json"

RAW_FILES = [
    (
        GOODREADS_RAW_DIR / "goodreads_interactions.csv",
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_interactions.csv",
        "goodreads_interactions.csv",
    ),
    (
        GOODREADS_RAW_DIR / "goodreads_books.json.gz",
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz",
        "goodreads_books.json.gz",
    ),
    (
        GOODREADS_RAW_DIR / "book_id_map.csv",
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/book_id_map.csv",
        "book_id_map.csv",
    ),
]

PROCESSED_FILES = [
    ("books.csv", BOOKS_PATH),
    ("users.csv", USERS_PATH),
    ("interactions_train.csv", INTERACTIONS_TRAIN_PATH),
    ("interactions_val.csv", INTERACTIONS_VAL_PATH),
    ("interactions_test.csv", INTERACTIONS_TEST_PATH),
]

QUERY_K_DISPLAY = 10
QUERY_K_RETRIEVE = 200

@st.cache_resource(show_spinner="Carregando modelo TF-IDF...")
def load_tfidf_model() -> TFIDFContentRecommender:
    model = TFIDFContentRecommender()
    model.fit()
    return model

def load_embedding_model() -> EmbeddingContentRecommender:
    """
    Carrega o modelo de embeddings e gera os vetores dos livros.

    Usa st.session_state para cachear o modelo durante a sessão
    e um st.progress para exibir o avanço dos batches.
    """
    if "embedding_model" in st.session_state:
        return st.session_state["embedding_model"]

    model = EmbeddingContentRecommender()

    progress_bar = st.progress(
        0,
        text="Gerando embeddings dos livros (Técnica 2)...",
    )

    def progress_callback(done_batches: int, total_batches: int) -> None:
        if total_batches <= 0:
            return

        percent = done_batches * 100 / total_batches

        progress_bar.progress(
            int(percent),
            text=(
                "Gerando embeddings dos livros (Técnica 2)... "
                f"{percent:.2f} %"
            ),
        )

    model.fit(progress_callback=progress_callback)
    progress_bar.empty()

    st.session_state["embedding_model"] = model
    return model

@st.cache_data(show_spinner=False)
def load_interactions_train() -> pd.DataFrame:
    return pd.read_csv(INTERACTIONS_TRAIN_PATH)

@st.cache_data
def load_books() -> pd.DataFrame:
    """Carrega o catálogo de livros de data/processed/books.csv."""
    df = pd.read_csv(BOOKS_PATH)
    return df

@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_user_history(user_id: int) -> pd.DataFrame:
    inter_train = load_interactions_train()
    books_df = load_books()

    hist = inter_train[inter_train["user_id"] == user_id]
    if hist.empty:
        return pd.DataFrame(columns=["book_id", "title"])

    hist = hist.merge(books_df[["book_id", "title"]], on="book_id", how="left")
    hist = hist.drop_duplicates(subset="book_id")
    return hist[["book_id", "title"]]

def show_user_history(user_id: int, max_books: int = 20) -> None:
    hist = get_user_history(user_id)
    if hist.empty:
        st.caption("Este usuário não possui histórico em `interactions_train.csv`.")
        return

    titles = (
        hist["title"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not titles:
        st.caption("Não foi possível encontrar títulos válidos para o histórico deste usuário.")
        return

    st.caption("Alguns livros já lidos por este usuário:")
    st.write(", ".join(titles[:max_books]))

def sample_random_user_id():
    df = load_interactions_train()
    if df.empty or "user_id" not in df.columns:
        return None
    return df["user_id"].sample(1).iloc[0]

def download_file_with_progress(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Falha ao baixar {label} ({url}): {e}")
        raise

    total = int(resp.headers.get("content-length", 0))
    progress = st.progress(0, text=f"{label} 0.00 %")
    bytes_downloaded = 0
    chunk_size = 1024 * 1024

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            if total > 0:
                bytes_downloaded += len(chunk)
                percent_float = bytes_downloaded * 100 / total
                percent_int = int(percent_float)
                if percent_int > 100:
                    percent_int = 100
                progress.progress(
                    percent_int,
                    text=f"{label} {percent_float:.2f} %",
                )

    progress.empty()

# Pós-processamento
SERIES_SUFFIXES = {
    "trilogy",
    "boxset",
    "box",
    "set",
    "collection",
    "complete",
    "movie",
    "companion",
    "guide",
    "official",
    "illustrated",
}

SERIES_GENERIC_TOKENS = {
    "chronicles",
    "chronicle",
    "trilogy",
    "saga",
    "series",
    "cycle",
    "duology",
    "quartet",
    "quintet",
    "book",
    "books",
    "novel",
    "novels",
    "story",
    "stories",
    "legend",
    "legends",
    "of",
    "the",
    "a",
    "an",
}

def check_and_prepare_dataset() -> None:
    """
    Verifica se os arquivos brutos e processados existem.
    - Se faltar raw: baixa de UCSD com barra de progresso.
    - Se faltar processed: roda pipeline make_splits.
    """
    st.markdown("#### Etapa 1 – Arquivos brutos (data/raw/goodreads)")

    GOODREADS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing_raw = []

    for path, url, label in RAW_FILES:
        if path.exists():
            st.success(f"{label} encontrado em `{path}`")
        else:
            st.warning(f"{label} não encontrado. Será baixado.")
            missing_raw.append((path, url, label))

    for path, url, label in missing_raw:
        download_file_with_progress(url, path, f"Baixando {label}")
        st.success(f"{label} baixado com sucesso.")

    st.markdown("#### Etapa 2 – Arquivos processados (data/processed)")

    missing_processed = []
    for label, path in PROCESSED_FILES:
        if path.exists():
            st.success(f"{label} encontrado em `{path}`")
        else:
            st.warning(f"{label} não encontrado em `{path}`")
            missing_processed.append((label, path))

    if not missing_processed:
        st.success("Todos os arquivos processados já existem. Conjunto de dados pronto para uso.")
        return

    st.info(
        "Arquivos processados incompletos. "
        "Executando pipeline de preparação (`make_splits`)."
    )

    try:
        from src.data_prep.make_splits import main as make_splits_main

        with st.spinner("Rodando pipeline de preparação de dados... (pode levar alguns minutos)"):
            make_splits_main()

        st.success("Pipeline concluído. Arquivos em `data/processed` gerados/atualizados.")
    except Exception as e:
        st.error(f"Falha ao executar o pipeline de preparação: {e}")

def _normalize_text_basic(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    tokens = t.split()
    ARTICLES = {"the", "a", "an"}
    tokens = [tok for tok in tokens if tok not in ARTICLES]
    return " ".join(tokens)

@st.cache_data(show_spinner=False)
def _books_with_normalized_title() -> pd.DataFrame:
    df = load_books().copy()
    df["normalized_title"] = df["title"].fillna("").astype(str).map(_normalize_text_basic)
    return df

def _parse_titles_from_query(query: str) -> list[str]:
    parts = re.split(r"[;\n]+", query)
    titles = [p.strip() for p in parts if p.strip()]
    return titles

def _map_input_titles_to_book_ids(input_titles: list[str]) -> tuple[list[str], set[int]]:
    books_df = _books_with_normalized_title()
    matched_titles: list[str] = []
    read_ids: set[int] = set()

    for raw in input_titles:
        norm = _normalize_text_basic(raw)
        if not norm:
            continue

        matches = books_df[books_df["normalized_title"] == norm]

        if matches.empty:
            tokens = set(norm.split())
            if tokens:
                mask = books_df["normalized_title"].apply(
                    lambda t: tokens.issubset(set(str(t).split()))
                )
                matches = books_df[mask]

        if matches.empty:
            continue 

        row = matches.iloc[0]
        read_ids.add(int(row["book_id"]))
        matched_titles.append(str(row["title"]))

    return matched_titles, read_ids

@st.cache_data(show_spinner=False)
def _load_user_histories_map() -> Dict[int, set[int]]:
    df = load_interactions_train()
    return df.groupby("user_id")["book_id"].apply(set).to_dict()

def _find_best_matching_user_from_books(read_ids: set[int]) -> Optional[int]:
    if not read_ids:
        return None

    user_histories = _load_user_histories_map()
    best_user_id = None
    best_overlap = 0

    for uid, books in user_histories.items():
        overlap = len(books & read_ids)
        if overlap > best_overlap:
            best_overlap = overlap
            best_user_id = uid

    if best_overlap == 0:
        return None
    return best_user_id

def _series_key_from_title(title: str) -> str:
    if not isinstance(title, str):
        title = "" if title is None else str(title)

    m = re.search(r"\(([^()]*?)#\d+", title)
    if m:
        raw_series = m.group(1)
        raw_series = raw_series.split(",")[0]
        key = _normalize_text_basic(raw_series)
    else:
        key = _normalize_text_basic(title)

    tokens = key.split()
    while tokens and tokens[-1] in SERIES_SUFFIXES:
        tokens.pop()
    if not tokens:
        tokens = key.split()
    return " ".join(tokens)

def _series_mentioned_in_query(series_key: str, query_norm: str) -> bool:
    if not series_key or not query_norm:
        return False

    if series_key in query_norm:
        return True

    q_tokens = set(query_norm.split())
    s_tokens = [t for t in series_key.split() if t not in SERIES_GENERIC_TOKENS]
    if not s_tokens:
        return False

    return any(tok in q_tokens for tok in s_tokens)

def _deduplicate_and_filter_by_series(
    recs: pd.DataFrame,
    k: int,
    query_norm: str,
) -> pd.DataFrame:
    if recs is None or recs.empty:
        return recs

    recs = recs.copy()
    if "title" not in recs.columns:
        return recs.head(k)

    recs["series_key"] = recs["title"].fillna("").astype(str).map(_series_key_from_title)

    if query_norm:
        mask_keep = ~recs["series_key"].apply(
            lambda s: _series_mentioned_in_query(s, query_norm)
        )
        recs = recs[mask_keep]

    if recs.empty:
        return recs

    if "score" not in recs.columns:
        recs = recs.drop_duplicates(subset="series_key", keep="first")
        return recs.head(k).drop(columns=["series_key"])

    recs = recs.sort_values("score", ascending=False)
    recs = recs.drop_duplicates(subset="series_key", keep="first")
    return recs.head(k).drop(columns=["series_key"])

def postprocess_query_recs(
    recs: pd.DataFrame,
    query: str,
    k: int = QUERY_K_DISPLAY,
) -> pd.DataFrame:
    query_norm = _normalize_text_basic(query)
    return _deduplicate_and_filter_by_series(recs, k=k, query_norm=query_norm)

def postprocess_history_recs(
    recs: pd.DataFrame,
    user_titles: list[str],
    user_read_ids: set[Any],
    k: int = 10,
) -> pd.DataFrame:
    """
    Pós-processa recomendações da Técnica 4 quando a consulta é gerada
    a partir do histórico do usuário.

    - Remove livros cujo book_id já foi lido pelo usuário.
    - Remove livros cuja série (series_key) já aparece no histórico.
    - Deduplica por série dentro do próprio ranking.
    """
    if recs is None or recs.empty:
        return recs

    recs = recs.copy()

    if "book_id" in recs.columns and user_read_ids:
        recs = recs[~recs["book_id"].isin(user_read_ids)]

    if recs.empty:
        return recs

    user_series_keys = {
        _series_key_from_title(t)
        for t in user_titles
        if isinstance(t, str) and t.strip()
    }

    recs["series_key"] = (
        recs["title"]
        .fillna("")
        .astype(str)
        .map(_series_key_from_title)
    )

    if user_series_keys:
        recs = recs[~recs["series_key"].isin(user_series_keys)]

    if recs.empty:
        return recs

    if "score" in recs.columns:
        recs = recs.sort_values("score", ascending=False)

    recs = recs.drop_duplicates(subset="series_key", keep="first")
    recs = recs.drop(columns=["series_key"], errors="ignore")

    cols = [c for c in ["book_id", "title", "score"] if c in recs.columns]
    if cols:
        recs = recs[cols]

    return recs.head(k)

def ensure_gemini_api_key() -> bool:
    """
    Garante que há uma chave de API do Gemini disponível.
    - Lê de st.session_state["gemini_api_key"]
    - Seta a variável de ambiente GEMINI_API_KEY_ENVVAR
    - Se não tiver, mostra erro e retorna False
    """
    api_key = st.session_state.get("gemini_api_key", "").strip()
    if not api_key:
        st.error(
            "Para usar a Técnica 4 (RAG + LLM), você precisa informar a chave da API "
            "do Gemini na seção de configuração."
        )
        return False

    os.environ[GEMINI_API_KEY_ENVVAR] = api_key
    return True

def build_llm_explanation_for_recs(query: str, recs_df: pd.DataFrame) -> str:
    """
    Gera o texto de explicação do Gemini a partir da MESMA lista de livros
    que aparece na tabela (já pós-processada).
    """
    if recs_df is None or recs_df.empty:
        return (
            "Não foi possível gerar explicação porque nenhuma recomendação foi "
            "encontrada para essa descrição."
        )

    lines = []
    lines.append("Você é um sistema de recomendação de livros para montar bibliografias.")
    lines.append("O usuário fez a seguinte descrição / pedido:")
    lines.append(f"\"{query}\"")
    lines.append("")
    lines.append("A seguir estão alguns livros recomendados (id, título, descrição):")

    max_books_for_llm = 5
    for idx, row in enumerate(recs_df.head(max_books_for_llm).itertuples(), start=1):
        desc = (getattr(row, "description", "") or "").replace("\n", " ")
        lines.append(
            f"{idx}. [book_id={row.book_id}] "
            f"Título: {row.title} | Descrição: {desc}"
        )

    lines.append("")
    lines.append(
        "Com base na consulta e nos livros acima, escreva em português uma explicação "
        "curta em formato de tópicos numerados, dizendo por que esses livros são boas "
        "recomendações para o usuário."
    )
    lines.append(
        "Regras importantes: "
        "1) Não repita o mesmo livro nem diferentes edições do mesmo livro; "
        "2) Se vários livros forem da mesma série, agrupe a série em um único tópico; "
        "3) Não invente títulos que não estejam na lista."
    )

    prompt = "\n".join(lines)
    return generate_llm_response(prompt)

# Helpers de visualização
def show_metrics_table(title: str, metrics: dict | None) -> None:
    """
    Mostra tabela de métricas em formato padrão.

    - Para T1/T2: usa as chaves 'precision@k', 'recall@k', 'ndcg@k' direto.
    - Para T4 (RAG retriever): achata o dicionário:
        metrics["metrics"]["precision"]["5"] -> "precision@5"
    - Ignora 'model', 'num_users' e 'top_k'.
    """
    st.subheader(title)

    if not metrics:
        st.info("Nenhuma métrica disponível para esta técnica.")
        return

    rows: list[dict] = []

    if isinstance(metrics, dict) and "metrics" in metrics and isinstance(metrics["metrics"], dict):
        nested = metrics["metrics"]

        for metric_name, per_k in nested.items():
            if not isinstance(per_k, dict):
                continue

            for k_str, value in per_k.items():
                label = f"{metric_name}@{k_str}"
                try:
                    val_float = float(value)
                    display_val = f"{val_float:.4f}"
                except (TypeError, ValueError):
                    display_val = str(value)

                rows.append({"métrica": label, "valor": display_val})

    else:
        for key, value in metrics.items():
            if key in {"model", "num_users", "top_k"}:
                continue

            try:
                val_float = float(value)
                display_val = f"{val_float:.4f}"
            except (TypeError, ValueError):
                display_val = str(value)

            rows.append({"métrica": key, "valor": display_val})

    if not rows:
        st.info("Nenhuma métrica numérica encontrada no JSON.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

def show_recs_table(recs: pd.DataFrame, max_rows: int = 10) -> None:
    if recs is None or recs.empty:
        st.info("Nenhuma recomendação encontrada.")
        return

    cols_to_show = [c for c in ["book_id", "title", "score"] if c in recs.columns]
    st.dataframe(recs[cols_to_show].head(max_rows), width="stretch")

def load_json_if_exists(path: Path) -> Optional[Dict]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def run_t3_from_text_history(query: str, k: int = QUERY_K_DISPLAY) -> None:
    raw_titles = _parse_titles_from_query(query)
    matched_titles, read_ids = _map_input_titles_to_book_ids(raw_titles)
    num_read = len(read_ids)

    if num_read < 4:
        st.warning(
            "Dados históricos insuficientes para realizar uma sugestão "
            "(informe pelo menos 4 livros que você já leu, separados por ';' ou por linha)."
        )
        if matched_titles:
            st.caption("Livros reconhecidos no catálogo:")
            st.write(", ".join(matched_titles))
        return

    best_user_id = _find_best_matching_user_from_books(read_ids)
    if best_user_id is None:
        st.warning(
            "Não encontramos um usuário do Goodreads com histórico compatível "
            "com esses livros. Tente adicionar outros títulos."
        )
        return

    st.caption(
        f"Seu perfil foi aproximado ao usuário `{best_user_id}` do Goodreads, "
        "que possui histórico semelhante."
    )

    try:
        base_k = max(k * 5, k + 20)

        recs = recommend_for_user_neural_cf(best_user_id, k=base_k)

        if "book_id" in recs.columns:
            recs = recs[~recs["book_id"].isin(read_ids)]

        recs = postprocess_history_recs(
            recs,
            user_titles=matched_titles,
            user_read_ids=read_ids,
            k=k,
        )

        show_recs_table(recs, max_rows=k)
    except Exception as e:
        st.error(
            f"Erro ao recomendar com Técnica 3 (FCN) a partir dos livros informados: {e}"
        )

def main() -> None:
    st.title("Sistema de Recomendação de Materiais Bibliográficos (Goodreads)")

    st.markdown(
        """
        Projeto de **Aprendizado Profundo / AM** – recomendação de livros
        usando o acervo Goodreads.

        **Técnicas consideradas:**

        - **Técnica 1 – TF-IDF por conteúdo (Term Frequency – Inverse Document Frequency)**  
          Representa cada livro como um vetor de pesos que mede quão importante é
          cada termo (palavra) no título e na descrição daquele livro em relação
          ao restante do acervo. A recomendação é feita por similaridade de
          cosseno entre vetores TF-IDF.

        - **Técnica 2 – Embeddings semânticos por conteúdo**  
          Usa modelos de linguagem (SentenceTransformers) para gerar embeddings
          densos a partir de título + descrição, capturando semelhanças semânticas
          mesmo quando as palavras não são idênticas. A recomendação é feita por
          similaridade de cosseno nesse espaço vetorial.

        - **Técnica 3 – FCN: Filtragem Colaborativa Neural**  
          Modelo de *Neural Collaborative Filtering* treinado diretamente sobre a
          matriz usuário–item (leituras/avaliações). O modelo aprende embeddings
          de usuários e de livros e estima a probabilidade de um usuário se
          interessar por cada livro.

        - **Técnica 4 – RAG + LLM (Retrieval-Augmented Generation)**  
          Combina um *retriever* baseado em embeddings (ChromaDB sobre o acervo
          Goodreads) com um modelo de linguagem (Gemini). O retriever recupera
          livros relevantes para consultas em linguagem natural e o LLM gera uma
          explicação textual organizando as recomendações.
        """
    )

    st.markdown("### Verificação do conjunto de dados")

    if st.button("Verificar conjunto de dados"):
        check_and_prepare_dataset()

    st.markdown("---")
    st.subheader("Configuração de API para LLM (Técnica 4)")

    default_key = st.session_state.get("gemini_api_key", "")
    api_key_input = st.text_input(
        "Gemini API key",
        value=default_key,
        type="password",
        help=(
            "Cole aqui a sua chave de API do Gemini. "
            "Ela será usada apenas nesta sessão para a Técnica 4 (RAG + LLM)."
        ),
    )
    st.session_state["gemini_api_key"] = api_key_input.strip()

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Coluna ESQUERDA – avaliação com dataset Goodreads
    with col_left:
        st.header("Avaliação com usuários do Goodreads")

        st.markdown("Selecione qual técnica avaliar:")

        btn_eval_t1 = st.button("Técnica 1 – TF-IDF", key="eval_t1")
        btn_eval_t2 = st.button("Técnica 2 – Embeddings", key="eval_t2")
        btn_eval_t3 = st.button("Técnica 3 – FCN", key="eval_t3")
        btn_eval_t4 = st.button("Técnica 4 – RAG + LLM", key="eval_t4")
        btn_eval_all = st.button("Rodar todas as técnicas e comparar", key="eval_all")

        st.markdown("### Resultados – avaliação offline")
        results_container = st.container()

        metrics_t1 = load_metrics(TFIDF_METRICS_PATH)
        metrics_t2 = load_metrics(EMB_METRICS_PATH)
        metrics_t3 = load_metrics(NEURAL_CF_METRICS_PATH)
        metrics_t4 = load_json_if_exists(RAG_METRICS_FILE)

        # T1
        if btn_eval_t1:
            results_container.empty()
            with results_container:
                show_metrics_table("Técnica 1 – TF-IDF", metrics_t1)

                st.markdown("#### Exemplo de recomendações (usuário aleatório)")
                user_id = sample_random_user_id()
                if user_id is None:
                    st.warning("Não foi possível amostrar um usuário.")
                else:
                    st.caption(f"Usuário de exemplo (treino): `{user_id}`")
                    show_user_history(user_id)
                    model = load_tfidf_model()
                    try:
                        recs = model.recommend_for_user(user_id, k=10)
                        show_recs_table(recs, max_rows=10)
                    except Exception as e:
                        st.error(
                            f"Erro ao gerar recomendações para o usuário {user_id}: {e}"
                        )

        # T2
        if btn_eval_t2:
            results_container.empty()
            with results_container:
                show_metrics_table("Técnica 2 – Embeddings", metrics_t2)

                st.markdown("#### Exemplo de recomendações (usuário aleatório)")
                user_id = sample_random_user_id()
                if user_id is None:
                    st.warning("Não foi possível amostrar um usuário.")
                else:
                    st.caption(f"Usuário de exemplo (treino): `{user_id}`")
                    show_user_history(user_id)
                    try:
                        with st.spinner(
                            "Gerando recomendações com embeddings... "
                            "(pode levar alguns segundos)"
                        ):
                            model = load_embedding_model()
                            recs = model.recommend_for_user(user_id, k=10)
                        show_recs_table(recs, max_rows=10)
                    except Exception as e:
                        st.error(
                            f"Erro ao gerar recomendações para o usuário {user_id}: {e}"
                        )

        # T3
        if btn_eval_t3:
            results_container.empty()
            with results_container:
                show_metrics_table("Técnica 3 – FCN (Neural CF)", metrics_t3)

                st.markdown("#### Exemplo de recomendações (usuário aleatório)")
                user_id = sample_random_user_id()
                if user_id is None:
                    st.warning("Não foi possível amostrar um usuário.")
                else:
                    st.caption(f"Usuário de exemplo (treino): `{user_id}`")
                    show_user_history(user_id)
                    try:
                        recs = recommend_for_user_neural_cf(user_id, k=10)
                        show_recs_table(recs, max_rows=10)
                    except Exception as e:
                        st.error(
                            f"Erro ao gerar recomendações (T3) para o usuário {user_id}: {e}"
                        )

        # T4
        if btn_eval_t4:
            results_container.empty()
            with results_container:
                st.markdown(
                    "#### Métricas – Técnica 4 (RAG, avaliando apenas o *retriever*)"
                )

                metrics_t4_local = None

                if metrics_t4 is None:
                    st.warning(
                        "Ainda não existe arquivo de métricas para a Técnica 4 "
                        "(results/rag_retriever_metrics.json)."
                    )
                    if st.button(
                        "Calcular métricas da Técnica 4 (pode demorar)",
                        key="btn_eval_t4_run",
                    ):
                        with st.spinner(
                            "Rodando avaliação offline da Técnica 4 "
                            "(pode levar alguns minutos)..."
                        ):
                            new_metrics_t4 = evaluate_rag_retriever()
                            save_metrics(new_metrics_t4, name="rag_retriever")
                        st.success(
                            "Métricas da Técnica 4 calculadas e salvas em "
                            "results/rag_retriever_metrics.json."
                        )
                        metrics_t4_local = new_metrics_t4
                else:
                    metrics_t4_local = metrics_t4

                if metrics_t4_local is not None:
                    show_metrics_table("Técnica 4 – RAG (retriever)", metrics_t4_local)
                    st.markdown("#### Exemplo de recomendações (usuário aleatório)")

                    if ensure_gemini_api_key():
                        user_id = sample_random_user_id()
                        if user_id is None:
                            st.warning("Não foi possível amostrar um usuário.")
                        else:
                            st.caption(f"Usuário de exemplo (treino): `{user_id}`")
                            show_user_history(user_id)

                            inter_train = load_interactions_train()
                            books_df = load_books()

                            hist_user = inter_train[inter_train["user_id"] == user_id]
                            user_read_ids = set(hist_user["book_id"].unique())

                            if hist_user.empty:
                                st.warning(
                                    "Não foi possível encontrar histórico para este usuário em "
                                    "`interactions_train.csv`. Tente clicar em 'Técnica 4' novamente."
                                )
                            else:
                                hist_merged = hist_user.merge(
                                    books_df, on="book_id", how="left"
                                )
                                user_titles = (
                                    hist_merged["title"]
                                    .dropna()
                                    .astype(str)
                                    .drop_duplicates()
                                    .tolist()
                                )

                                if not user_titles:
                                    st.warning(
                                        "O usuário de exemplo não possui títulos válidos "
                                        "para montar a consulta textual."
                                    )
                                else:
                                    sample_titles = user_titles[:5]
                                    query_text = (
                                        "I have recently read the following books: "
                                        + "; ".join(sample_titles)
                                        + ". Please recommend other similar books."
                                    )

                                    st.caption(
                                        "Consulta textual gerada automaticamente a partir do histórico:"
                                    )
                                    st.markdown(f"> {query_text}")

                                    try:
                                        with st.spinner(
                                            "Consultando a Técnica 4 (RAG + LLM) para este usuário..."
                                        ):
                                            rag_result = recommend_with_rag_for_query(
                                                query_text, k=50
                                            )

                                        recs = pd.DataFrame(
                                            rag_result.get("recommended_books", [])
                                        )

                                        if recs.empty:
                                            st.warning(
                                                "A Técnica 4 não retornou recomendações para este usuário."
                                            )
                                        else:
                                            if "score" not in recs.columns:
                                                if "similarity" in recs.columns:
                                                    recs.rename(
                                                        columns={"similarity": "score"},
                                                        inplace=True,
                                                    )
                                                else:
                                                    recs["score"] = None

                                            recs = postprocess_history_recs(
                                                recs,
                                                user_titles=user_titles,
                                                user_read_ids=user_read_ids,
                                                k=10,
                                            )

                                            if recs.empty:
                                                st.info(
                                                    "Todas as recomendações da Técnica 4 para este usuário "
                                                    "eram livros que ele já leu ou pertenciam a séries já lidas."
                                                )
                                            else:
                                                show_recs_table(recs, max_rows=10)
                                    except Exception as e:
                                        st.error(
                                            f"Erro ao obter recomendações com a Técnica 4 (RAG): {e}"
                                        )

        # Rodar todas
        if btn_eval_all:
            results_container.empty()
            with results_container:
                with st.spinner(
                    "Rodando todas as técnicas (pode levar alguns segundos na primeira "
                    "execução, especialmente para carregar os embeddings da Técnica 2)..."
                ):
                    st.subheader("Comparação de métricas entre técnicas")

                    rows: list[dict] = []

                    # T1: TF-IDF
                    if metrics_t1 is not None:
                        row1: dict = {
                            "técnica": "T1 – TF-IDF",
                            "num_users": metrics_t1.get("num_users"),
                        }
                        for k, v in metrics_t1.items():
                            if k in {"model", "num_users"}:
                                continue
                            row1[k] = float(v)
                        rows.append(row1)

                    # T2: Embeddings
                    if metrics_t2 is not None:
                        row2: dict = {
                            "técnica": "T2 – Embeddings",
                            "num_users": metrics_t2.get("num_users"),
                        }
                        for k, v in metrics_t2.items():
                            if k in {"model", "num_users"}:
                                continue
                            row2[k] = float(v)
                        rows.append(row2)

                    # T3: FCN
                    if metrics_t3 is not None:
                        row3: dict = {
                            "técnica": "T3 – FCN",
                            "num_users": metrics_t3.get("num_users"),
                        }
                        nested3 = metrics_t3.get("metrics", {})
                        if isinstance(nested3, dict):
                            for metric_name, per_k in nested3.items():
                                if not isinstance(per_k, dict):
                                    continue
                                for k_str, value in per_k.items():
                                    label = f"{metric_name}@{k_str}"
                                    try:
                                        row3[label] = float(value)
                                    except (TypeError, ValueError):
                                        row3[label] = None
                        rows.append(row3)

                    # T4: RAG
                    if metrics_t4 is not None:
                        row4: dict = {
                            "técnica": "T4 – RAG (retriever)",
                            "num_users": metrics_t4.get("num_users"),
                        }

                        nested = metrics_t4.get("metrics", {})
                        if isinstance(nested, dict):
                            for metric_name, per_k in nested.items():
                                if not isinstance(per_k, dict):
                                    continue
                                for k_str, value in per_k.items():
                                    label = f"{metric_name}@{k_str}"
                                    try:
                                        row4[label] = float(value)
                                    except (TypeError, ValueError):
                                        row4[label] = None

                        rows.append(row4)

                    if rows:
                        df = pd.DataFrame(rows).set_index("técnica")
                        st.dataframe(
                            df.style.format({"valor": "{:.6f}"}), width="stretch"
                        )
                    else:
                        st.warning(
                            "Nenhuma métrica encontrada. "
                            "Certifique-se de rodar `python -m src.evaluation.evaluate_models` antes."
                        )

                    st.markdown(
                        "#### Recomendações para um mesmo usuário (comparação qualitativa)"
                    )
                    user_id = sample_random_user_id()
                    if user_id is None:
                        st.warning("Não foi possível amostrar um usuário.")
                    else:
                        st.caption(f"Usuário de exemplo (treino): `{user_id}`")
                        show_user_history(user_id)
                        tabs = st.tabs(
                            [
                                "T1 – TF-IDF",
                                "T2 – Embeddings",
                                "T3 – FCN",
                                "T4 – RAG + LLM",
                            ]
                        )

                        with tabs[0]:
                            model = load_tfidf_model()
                            try:
                                recs = model.recommend_for_user(user_id, k=10)
                                show_recs_table(recs, max_rows=10)
                            except Exception as e:
                                st.error(
                                    f"Erro ao gerar recomendações (T1) para {user_id}: {e}"
                                )

                        with tabs[1]:
                            model = load_embedding_model()
                            try:
                                recs = model.recommend_for_user(user_id, k=10)
                                show_recs_table(recs, max_rows=10)
                            except Exception as e:
                                st.error(
                                    f"Erro ao gerar recomendações (T2) para {user_id}: {e}"
                                )

                        with tabs[2]:
                            st.subheader("Técnica 3 – Neural CF")

                            try:
                                recs = recommend_for_user_neural_cf(user_id, k=10)
                                show_recs_table(recs, max_rows=10)
                            except Exception as e:
                                st.error(
                                    f"Erro ao gerar recomendações (T3) para {user_id}: {e}"
                                )

                        with tabs[3]:
                            st.markdown(
                                "#### Técnica 4 – RAG (recomendações geradas a partir do histórico)"
                            )

                            inter_train = load_interactions_train()
                            books_df = load_books()

                            hist_user = inter_train[inter_train["user_id"] == user_id]
                            user_read_ids = set(hist_user["book_id"].unique())

                            if hist_user.empty:
                                st.warning(
                                    "Não foi possível encontrar histórico para este usuário em "
                                    "`interactions_train.csv`. Tente clicar em "
                                    "'Rodar todas as técnicas e comparar' novamente."
                                )
                            else:
                                hist_merged = hist_user.merge(
                                    books_df, on="book_id", how="left"
                                )
                                user_titles = (
                                    hist_merged["title"]
                                    .dropna()
                                    .astype(str)
                                    .drop_duplicates()
                                    .tolist()
                                )

                                if not user_titles:
                                    st.warning(
                                        "O usuário de exemplo não possui títulos válidos "
                                        "para montar a consulta textual."
                                    )
                                else:
                                    sample_titles = user_titles[:5]
                                    query_text = (
                                        "I have recently read the following books: "
                                        + "; ".join(sample_titles)
                                        + ". Please recommend other similar books."
                                    )

                                    st.caption(
                                        "Consulta textual gerada automaticamente a partir do histórico:"
                                    )
                                    st.markdown(f"> {query_text}")

                                    if ensure_gemini_api_key():
                                        try:
                                            with st.spinner(
                                                "Consultando a Técnica 4 (RAG + LLM) para este usuário..."
                                            ):
                                                rag_result = recommend_with_rag_for_query(
                                                    query_text, k=50
                                                )

                                            recs = pd.DataFrame(
                                                rag_result.get("recommended_books", [])
                                            )

                                            if recs.empty:
                                                st.warning(
                                                    "A Técnica 4 não retornou recomendações para este usuário."
                                                )
                                            else:
                                                if "score" not in recs.columns:
                                                    if "similarity" in recs.columns:
                                                        recs.rename(
                                                            columns={
                                                                "similarity": "score"
                                                            },
                                                            inplace=True,
                                                        )
                                                    else:
                                                        recs["score"] = None

                                                recs = postprocess_history_recs(
                                                    recs,
                                                    user_titles=user_titles,
                                                    user_read_ids=user_read_ids,
                                                    k=10,
                                                )

                                                if recs.empty:
                                                    st.info(
                                                        "Todas as recomendações da Técnica 4 para este usuário "
                                                        "eram livros que ele já leu ou pertenciam a séries já lidas."
                                                    )
                                                else:
                                                    show_recs_table(recs, max_rows=10)
                                                    st.caption(
                                                        "As recomendações da Técnica 4 foram geradas a partir "
                                                        "de uma consulta textual construída com base no "
                                                        "histórico desse mesmo usuário."
                                                    )
                                        except Exception as e:
                                            st.error(
                                                f"Erro ao obter recomendações com a Técnica 4 (RAG): {e}"
                                            )
    
    # Coluna DIREITA – busca por texto / descrição
    with col_right:
        st.header("Busca por texto / descrição (modo usuário final)")
        st.markdown(
            """
            Você pode usar esta área de duas formas principais:

            - **T1, T2 e T4** – escreva em inglês uma descrição do tipo de livro que procura  
            (ex.: tema, clima da história, referências a outros livros/autores).

            - **T3 – FCN (Filtragem Colaborativa Neural)** – liste alguns livros que você já leu
            (em inglês), **um por linha ou separados por `;`**.  
            Se **4 ou mais livros forem reconhecidos no catálogo**, o sistema aproxima seu perfil
            de um usuário do Goodreads com histórico semelhante e gera recomendações a partir disso.  
            Se menos de 4 livros forem reconhecidos, será exibida a mensagem de
            **“dados históricos insuficientes para realizar uma sugestão”**.
            """
        )

        query = st.text_area(
            "Descreva o livro que você leu ou o tipo de livro que procura:",
            height=140,
        )

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            btn_q_t1 = st.button("Recomendar com T1 – TF-IDF", key="q_t1")
            btn_q_t3 = st.button("Recomendar com T3 – FCN", key="q_t3")
        with col_q2:
            btn_q_t2 = st.button("Recomendar com T2 – Embeddings", key="q_t2")
            btn_q_t4 = st.button("Recomendar com T4 – RAG + LLM", key="q_t4")

        btn_q_all = st.button("Rodar todas as técnicas (texto livre)", key="q_all")

        st.markdown("### Resultados – recomendações a partir da descrição")

        any_query_button = btn_q_t1 or btn_q_t2 or btn_q_t3 or btn_q_t4 or btn_q_all

        if any_query_button:
            if not query or not query.strip():
                st.warning("Digite uma descrição ou título antes de rodar as recomendações.")
            else:
                if btn_q_t1:
                    st.subheader("Técnica 1 – TF-IDF (conteúdo)")
                    model = load_tfidf_model()
                    try:
                        recs = model.recommend_for_query(query, k=QUERY_K_RETRIEVE)
                        recs = postprocess_query_recs(recs, query, k=QUERY_K_DISPLAY)
                        show_recs_table(recs, max_rows=QUERY_K_DISPLAY)
                    except Exception as e:
                        st.error(f"Erro ao recomendar com TF-IDF: {e}")

                if btn_q_t2:
                    st.subheader("Técnica 2 – Embeddings (conteúdo)")
                    try:
                        with st.spinner("Gerando recomendações com embeddings... (pode levar alguns segundos)"):
                            model = load_embedding_model()
                            recs = model.recommend_for_query(query, k=QUERY_K_RETRIEVE)
                            recs = postprocess_query_recs(recs, query, k=QUERY_K_DISPLAY)
                        show_recs_table(recs, max_rows=QUERY_K_DISPLAY)
                    except Exception as e:
                        st.error(f"Erro ao recomendar com embeddings: {e}")

                if btn_q_t3:
                    st.subheader("Técnica 3 – FCN (Filtragem Colaborativa Neural)")
                    run_t3_from_text_history(query, k=QUERY_K_DISPLAY)

                if btn_q_t4:
                    st.subheader("Técnica 4 – RAG + LLM")
                    if ensure_gemini_api_key():
                        try:
                            result_t4 = recommend_with_rag_for_query(
                                query,
                                k=QUERY_K_RETRIEVE,
                                return_explanation=False,
                            )

                            recs_t4 = pd.DataFrame(result_t4["recommended_books"])

                            recs_t4 = postprocess_query_recs(
                                recs_t4,
                                query,
                                k=QUERY_K_DISPLAY,
                            )
                            show_recs_table(recs_t4, max_rows=QUERY_K_DISPLAY)

                            with st.expander(
                                "Explicação gerada pelo LLM (Gemini)",
                                expanded=False,
                            ):
                                explanation = build_llm_explanation_for_recs(query, recs_t4)
                                st.write(explanation)

                        except Exception as e:
                            st.error(
                                f"Erro ao recomendar com Técnica 4 (RAG + LLM): {e}"
                            )

                if btn_q_all:
                    tabs = st.tabs(
                        [
                            "T1 – TF-IDF",
                            "T2 – Embeddings",
                            "T3 – FCN",
                            "T4 – RAG + LLM",
                        ]
                    )

                    with tabs[0]:
                        st.subheader("Técnica 1 – TF-IDF")
                        model = load_tfidf_model()
                        try:
                            recs = model.recommend_for_query(query, k=QUERY_K_RETRIEVE)
                            recs = postprocess_query_recs(recs, query, k=QUERY_K_DISPLAY)
                            show_recs_table(recs, max_rows=QUERY_K_DISPLAY)
                        except Exception as e:
                            st.error(f"Erro ao recomendar com TF-IDF: {e}")

                    with tabs[1]:
                        st.subheader("Técnica 2 – Embeddings")
                        model = load_embedding_model()
                        try:
                            recs = model.recommend_for_query(query, k=QUERY_K_RETRIEVE)
                            recs = postprocess_query_recs(recs, query, k=QUERY_K_DISPLAY)
                            show_recs_table(recs, max_rows=QUERY_K_DISPLAY)
                        except Exception as e:
                            st.error(f"Erro ao recomendar com embeddings: {e}")

                    with tabs[2]:
                        st.subheader("Técnica 3 – FCN")
                        run_t3_from_text_history(query, k=QUERY_K_DISPLAY)

                    with tabs[3]:
                        st.subheader("Técnica 4 – RAG + LLM")

                        if ensure_gemini_api_key():
                            try:
                                result_t4 = recommend_with_rag_for_query(
                                    query,
                                    k=QUERY_K_RETRIEVE,
                                    return_explanation=False,
                                )

                                recs_t4 = pd.DataFrame(result_t4["recommended_books"])
                                recs_t4 = postprocess_query_recs(
                                    recs_t4,
                                    query,
                                    k=QUERY_K_DISPLAY,
                                )

                                show_recs_table(recs_t4, max_rows=QUERY_K_DISPLAY)

                                with st.expander(
                                    "Explicação gerada pelo LLM (Gemini)",
                                    expanded=False,
                                ):
                                    explanation = build_llm_explanation_for_recs(query, recs_t4)
                                    st.write(explanation)

                            except Exception as e:
                                st.error(
                                    f"Erro ao recomendar com Técnica 4 (RAG + LLM): {e}"
                                )

if __name__ == "__main__":
    main()
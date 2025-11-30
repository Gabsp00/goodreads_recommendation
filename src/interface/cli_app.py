"""
Interface em linha de comando para testar o sistema de recomendação
a partir de uma descrição em linguagem natural.

Uso:

    (recsys-am) $ python -m src.interface.cli_app
"""

from __future__ import annotations

import re
import pandas as pd

from src.models.tfidf_content import TFIDFContentRecommender
from src.models.embedding_content import EmbeddingContentRecommender

K_DISPLAY = 10
K_RETRIEVE = 200

# Sufixos genéricos para remover do "nome de série"
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

# Helpers de normalização
def _normalize_text_basic(text: str) -> str:
    """
    Normalização básica usada tanto para:
    - construir series_key
    - normalizar a query do usuário

    Passos:
    - minúsculas
    - mantém apenas letras/números/espaços
    - remove artigos muito comuns ("the", "a", "an")
    - comprime espaços
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    tokens = t.split()

    ARTICLES = {"the", "a", "an"}
    tokens = [tok for tok in tokens if tok not in ARTICLES]

    t = " ".join(tokens)
    return t

def _series_key_from_title(title: str) -> str:
    """
    Tenta extrair uma chave de SÉRIE a partir do título.

    Estratégia:
    1) Se houver padrão "(Algo, #N)" ou "(Algo #N)", usa "Algo" como nome da série.
       Ex: "Catching Fire (Hunger Games, #2)" -> "hunger games"
           "Divergent (Divergent, #1)"        -> "divergent"
    2) Caso contrário, usa o título completo normalizado, mas removendo
       sufixos genéricos como "trilogy", "boxset", "collection" etc.
       Ex: "The Hunger Games Trilogy Boxset" -> "the hunger games"
    """
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
    """
    Decide se uma série foi mencionada na query.

    - Se o series_key inteiro aparecer na query, bloqueia.
    - Senão, se qualquer token "importante" da série aparecer na query,
      também considera mencionado (ignora tokens genéricos).
    """
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
    """
    - Cria uma coluna 'series_key' para cada livro.
    - Remove TODAS as séries cuja series_key apareça na query normalizada
      (ex.: 'hunger games' na query -> nenhum livro dessa série aparece).
    - Deduplica por series_key (mantém o de maior score).
    - Retorna top-k.
    """
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

def _print_recommendations(
    title: str,
    recs: pd.DataFrame,
    query: str,
    k: int = K_DISPLAY,
) -> None:
    """
    Imprime as top-k recomendações de um DataFrame com colunas:
    - book_id
    - title
    - (opcional) score
    """
    print(f"\n=== {title} ===")
    if recs is None or recs.empty:
        print("Nenhuma recomendação encontrada.")
        return

    query_norm = _normalize_text_basic(query)
    recs = _deduplicate_and_filter_by_series(recs, k=k, query_norm=query_norm)

    if recs is None or recs.empty:
        print("Nenhuma recomendação após filtrar séries já mencionadas.")
        return

    cols = list(recs.columns)
    has_score = "score" in cols

    for i, row in enumerate(recs.itertuples(), start=1):
        book_id = getattr(row, "book_id", "N/A")
        book_title = getattr(row, "title", "(sem título)")
        if has_score:
            score = getattr(row, "score", None)
            if score is not None:
                print(f"{i:2d}. [{book_id}] {book_title} (score: {score:.4f})")
            else:
                print(f"{i:2d}. [{book_id}] {book_title}")
        else:
            print(f"{i:2d}. [{book_id}] {book_title}")

def main() -> None:
    print("==============================================")
    print("  CLI - Sistema de Recomendação de Livros")
    print("  Técnicas 1 (TF-IDF) e 2 (Embeddings)")
    print("==============================================")
    print("Carregando modelos e catálogo de livros...")
    print("Isso pode demorar um pouco na primeira execução.")

    tfidf_rec = TFIDFContentRecommender()
    tfidf_rec.fit()

    emb_rec = EmbeddingContentRecommender()
    emb_rec.fit()

    print("\nModelos carregados! Agora você pode digitar descrições de livros.")
    print("Dica: como o catálogo é em inglês, consultas em inglês tendem a funcionar melhor.")
    print("Digite 'sair' (ou 'exit'/'quit') para encerrar.\n")

    while True:
        try:
            query = input(
                "Digite uma descrição / tema de livro\n"
                "(ou 'sair' para encerrar):\n> "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando CLI. Até mais!")
            break

        if not query:
            continue

        if query.lower() in {"sair", "exit", "quit"}:
            print("Encerrando CLI. Até mais!")
            break

        print("\nGerando recomendações para sua descrição...")

        try:
            recs_tfidf = tfidf_rec.recommend_for_query(query, k=K_RETRIEVE)
        except Exception as e:
            print(f"[ERRO] TF-IDF falhou ao gerar recomendações: {repr(e)}")
            recs_tfidf = None

        try:
            recs_emb = emb_rec.recommend_for_query(query, k=K_RETRIEVE)
        except Exception as e:
            print(f"[ERRO] Embeddings falharam ao gerar recomendações: {repr(e)}")
            recs_emb = None

        _print_recommendations(
            "Técnica 1 - TF-IDF (conteúdo)",
            recs_tfidf,
            query=query,
            k=K_DISPLAY,
        )
        _print_recommendations(
            "Técnica 2 - Embeddings semânticos (conteúdo)",
            recs_emb,
            query=query,
            k=K_DISPLAY,
        )

        print("\n----------------------------------------------\n")

if __name__ == "__main__":
    main()
"""
Técnica 4: RAG + LLM para recomendação explicada.

Fluxo usado na interface (parte da direita – texto livre):

1. Receber um texto de consulta em linguagem natural.
2. Usar o LLM para reescrever essa descrição em uma query curta
   de busca (palavras-chave, em inglês).
3. Consultar o vector_store com essa query reescrita para recuperar
   N livros candidatos.
4. Montar um prompt com o contexto dos livros candidatos.
5. Chamar o LLM (via llm_client) para gerar uma explicação.
"""

from typing import List, Dict, Any

from src.rag.vector_store import query_vector_store
from src.rag.llm_client import generate_llm_response

def _rewrite_query_with_llm(original_query: str) -> str:
    """
    Usa o LLM para transformar a descrição em linguagem natural
    em uma query curta de busca (palavras-chave).
    """
    if not original_query or not original_query.strip():
        return original_query

    prompt_lines: List[str] = []
    prompt_lines.append(
        "Você recebe descrições em linguagem natural sobre livros e precisa "
        "convertê-las em uma consulta curta de busca, em INGLÊS."
    )
    prompt_lines.append(
        "Regra: responda com UMA ÚNICA LINHA contendo de 5 a 12 termos, "
        "separados por vírgulas. Use gêneros literários, temas, e nomes de "
        "autores ou séries quando forem importantes. Não explique nada, "
        "não use frases completas, não traduza de volta para português."
    )
    prompt_lines.append("")
    prompt_lines.append("Descrição do usuário:")
    prompt_lines.append(original_query.strip())

    prompt = "\n".join(prompt_lines)

    try:
        raw = generate_llm_response(prompt) or ""
    except Exception:
        return original_query

    rewritten = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if not rewritten:
        return original_query

    return rewritten

def recommend_with_rag_for_query(
    query: str,
    k: int = 10,
    return_explanation: bool = True,
) -> Dict[str, Any]:
    """
    Função de recomendação com RAG para uma consulta textual.

    Parâmetros
    ----------
    query : str
        Descrição em linguagem natural fornecida pelo usuário.
    k : int
        Número de livros a recuperar do vector store.
    return_explanation : bool, default True
        Se True, também chama o LLM para gerar uma explicação textual.
        Se False, apenas retorna os livros recomendados; o chamador
        pode gerar a explicação separadamente.

    Retorno
    -------
        {
          "recommended_books": [
              { "book_id": ..., "title": ..., "description": ..., "score": ... },
              ...
          ],
          "llm_explanation": str | None
        }
    """

    rewritten_query = _rewrite_query_with_llm(query)

    candidates = query_vector_store(rewritten_query, k=k)

    lines: List[str] = []
    lines.append("Você é um sistema de recomendação de livros para montar bibliografias.")
    lines.append("O usuário fez a seguinte descrição / pedido original:")
    lines.append(f"\"{query}\"")
    lines.append("")
    lines.append("A seguir, uma lista de livros candidatos (id, título, descrição):")

    for idx, c in enumerate(candidates, start=1):
        meta = c.get("metadata", {}) or {}
        lines.append(
            f"{idx}. [book_id={c['book_id']}] "
            f"Título: {meta.get('title', '')} "
            f"Descrição: {meta.get('description', '')}"
        )

    lines.append("")
    lines.append(
        "Com base na descrição original do usuário e nos livros acima, "
        "explique (em português) por que esses livros são boas "
        "recomendações. Se fizer sentido, destaque quais seriam os mais "
        "importantes para uma bibliografia inicial."
    )

    prompt = "\n".join(lines)

    explanation = None
    if return_explanation:
        explanation = generate_llm_response(prompt)

    rec_books: List[Dict[str, Any]] = []
    for c in candidates:
        meta = c.get("metadata", {}) or {}
        rec_books.append(
            {
                "book_id": c["book_id"],
                "score": c.get("score"),
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
            }
        )

    return {
        "recommended_books": rec_books,
        "llm_explanation": explanation,
    }


if __name__ == "__main__":
    from src.rag.vector_store import build_vector_store

    build_vector_store()
    result = recommend_with_rag_for_query(
        "I loved The Hunger Games trilogy and I want similar dystopian YA books.",
        k=5,
    )
    print("\n=== EXPLICAÇÃO ===\n")
    print(result["llm_explanation"])
    print("\n=== LIVROS RECOMENDADOS ===")
    for b in result["recommended_books"]:
        print(b["book_id"], b["title"], b["score"])
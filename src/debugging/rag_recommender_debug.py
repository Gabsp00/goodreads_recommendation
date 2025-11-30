from src.rag.rag_recommender import recommend_with_rag_for_query

def main() -> None:
    print("=== Debug da Técnica 4 (RAG + LLM) ===")
    print("Digite uma descrição do que você quer ler.")
    print("Pressione ENTER em branco para sair.\n")

    query = input("Consulta:\n> ").strip()
    if not query:
        print("Nenhuma consulta informada. Encerrando.")
        return

    result = recommend_with_rag_for_query(query, k=10)

    print("\n=== EXPLICAÇÃO DO GEMINI ===")
    print(result["llm_explanation"])

    print("\n=== LIVROS RECOMENDADOS ===")
    for i, b in enumerate(result["recommended_books"], start=1):
        print(f"{i}. [{b['book_id']}] {b['title']} (score: {b['score']:.4f})")

if __name__ == "__main__":
    main()
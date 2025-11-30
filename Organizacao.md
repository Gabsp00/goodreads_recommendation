# Organização do projeto

Estrutura de pastas do repositório:

recsys_biblioteca/
├─ config.py                     # caminhos, hiperparâmetros e constantes globais
├─ requirements.txt              # dependências Python (pip)
├─ Organizacao.md                # este arquivo de documentação
│
├─ data/
│  ├─ raw/
│  │   └─ goodreads/             # arquivos brutos do dataset UCSD Goodreads (NÃO versionados)
│  │        # interações completas (≈ 4 GB)
│  │        #   wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_interactions.csv
│  │        #
│  │        # metadados de livros (≈ 1.9 GB)
│  │        #   wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz
│  │        #
│  │        # mapa de IDs de livros
│  │        #   wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/book_id_map.csv
│  │
│  ├─ processed/
│  │   ├─ books.csv              # catálogo final de livros (já limpo)
│  │   ├─ users.csv              # usuários filtrados (após critérios mínimos)
│  │   ├─ interactions_train.csv # interações para treino
│  │   ├─ interactions_val.csv   # interações para validação
│  │   ├─ interactions_test.csv  # interações para teste
│  │   └─ vector_store/          # artefatos do ChromaDB (índice vetorial da Técnica 4)
│  │        └─ ...               # arquivos internos do banco vetorial (chroma.sqlite3, *.bin, etc.)
│  │
│  └─ models/
│      └─ neural_cf.pt           # pesos treinados do modelo da Técnica 3 – FCN (Neural CF)
│
├─ results/
│  ├─ tfidf_metrics.json               # métricas da Técnica 1 – TF-IDF
│  ├─ embedding_metrics.json           # métricas da Técnica 2 – Embeddings
│  ├─ neural_cf_metrics.json           # métricas da Técnica 3 – FCN
│  └─ rag_retriever_metrics.json       # métricas do retriever da Técnica 4 – RAG
│
├─ src/
│  ├─ __init__.py
│  │
│  ├─ data_prep/
│  │   ├─ __init__.py
│  │   ├─ load_goodreads.py      # leitura do UCSD Goodreads bruto + filtros + geração de books/users/interactions
│  │   ├─ make_splits.py         # criação dos splits train/val/test a partir das interações filtradas
│  │   └─ preprocess_text.py     # limpeza/normalização de textos (títulos, descrições, etc.)
│  │
│  ├─ models/
│  │   ├─ __init__.py
│  │   ├─ tfidf_content.py       # Técnica 1 – recomendação baseada em TF-IDF de conteúdo
│  │   ├─ embedding_content.py   # Técnica 2 – recomendação com embeddings semânticos
│  │   └─ neural_cf.py           # Técnica 3 – FCN (Neural Collaborative Filtering)
│  │
│  ├─ rag/
│  │   ├─ __init__.py
│  │   ├─ vector_store.py        # construção e consulta ao banco vetorial (ChromaDB)
│  │   ├─ llm_client.py          # wrapper para provedores de LLM (OpenAI, Gemini) usando variáveis de ambiente
│  │   └─ rag_recommender.py     # pipeline da Técnica 4 – RAG + LLM (reordenação + explicação das recomendações)
│  │
│  ├─ evaluation/
│  │   ├─ __init__.py
│  │   ├─ metrics.py             # funções auxiliares de métricas (precision@k, recall@k, nDCG, etc.)
│  │   └─ evaluate_models.py     # avaliação offline das técnicas 1, 2, 3 e retriever da 4
│  │
│  ├─ interface/
│  │   ├─ __init__.py
│  │   ├─ cli_app.py             # interface de linha de comando para experimentos
│  │   └─ streamlit_app.py       # interface web final (dashboard da disciplina em Streamlit)
│  │
│  └─ debugging/
│      ├─ debug.py               # scripts de inspeção geral (dados/modelos)
│      ├─ debug_rag.py           # testes de desenvolvimento da Técnica 4 – RAG
│      └─ rag_recommender_debug.py
│
└─ __pycache__/                  # diretórios gerados automaticamente pelo Python (ignorados no Git)

# Sistema de Recomendação de Livros com Técnicas de Aprendizado de Máquina

Este repositório contém o projeto desenvolvido para a disciplina de **Aprendizado Profundo** do  
**Programa de Pós-Graduação em Ciência da Computação (PPGCC) da Unesp**  
([site do PPGCC](https://www.ibilce.unesp.br/#!/pos-graduacao/programas-de-pos-graduacao/ciencia-da-computacao/apresentacao/)),  
ministrada pelo **Prof. Dr. Denis Henrique Pinheiro Salvadeo**.

---

## Objetivo do projeto

Implementar e comparar diferentes técnicas de recomendação de livros usando o acervo do Goodreads:

- **Técnica 1 – TF-IDF por conteúdo**
- **Técnica 2 – Embeddings semânticos por conteúdo**
- **Técnica 3 – FCN (Filtragem Colaborativa Neural)**
- **Técnica 4 – RAG + LLM (Retrieval-Augmented Generation)**

A interface final é um dashboard em **Streamlit** que permite:

- avaliação offline das técnicas com usuários reais do Goodreads (métricas @k);
- comparação qualitativa de recomendações para um mesmo usuário;
- modo “usuário final” com busca por descrição ou lista de livros lidos.

---

## Como reproduzir o ambiente

Pré-requisitos:

- Python 3.10+
- Git
- (Opcional) ambiente virtual (`venv`, `conda`, etc.)

1. Clonar este repositório:

   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   cd SEU_REPOSITORIO
   ```

2. (Opcional) Criar e ativar um ambiente virtual.

3. Instalar as dependências:

   ```bash
   pip install -r requirements.txt
   ```

---

## Como executar a interface (Streamlit)

Na raiz do projeto, executar:

```bash
streamlit run src/interface/streamlit_app.py
```

Isso abre o dashboard com:

- verificação do conjunto de dados;
- configuração da API para a Técnica 4 (RAG + LLM);
- avaliação offline das técnicas 1–4;
- modo de busca por texto / descrição (usuário final).

---

## Organização do projeto

A organização detalhada de pastas e arquivos está descrita em:

- Organizacao.md

Esse arquivo documenta a estrutura de `data/`, `src/`, `results/` e os principais scripts do projeto.

---

## Observação sobre arquivos grandes (Git LFS)

Este repositório utiliza **Git Large File Storage (Git LFS)** para armazenar alguns
arquivos essenciais, como:

- `data/processed/interactions_train.csv`
- `data/processed/vector_store/chroma.sqlite3`

Para clonar o projeto com todos esses arquivos funcionando, é recomendado:

```bash
# instalar o Git LFS (exemplo em sistemas baseados em Debian/Ubuntu)
sudo apt install git-lfs
git lfs install

# clonar o repositório normalmente
git clone https://github.com/Gabsp00/goodreads_recommendation.git
cd goodreads_recommendation
```

Sem o Git LFS, os arquivos grandes aparecerão apenas como “ponteiros” de texto e
as Técnicas 3 (FCN) e 4 (RAG) podem não funcionar corretamente.

---

## Configurações dos modelos e hiperparâmetros

### Técnica 2 – Embeddings semânticos por conteúdo

| Configuração                 | Detalhe                         |
|-----------------------------|---------------------------------|
| Modelo                      | *all-mpnet-base-v2*             |
| Dimensão do Embedding (dim) | 768                             |
| `batch_size`                | 128                             |
| Normalização de Embeddings  | `True`                          |
| Perfil do Usuário           | Média dos embeddings dos itens  |

---

### Técnica 3 – FCN (Filtragem Colaborativa Neural)

| Configuração              | Detalhe                        |
|---------------------------|--------------------------------|
| Dimensão dos Embeddings   | 64 (usuário e item)            |
| Arquitetura MLP           | 128 → 64 → 1                   |
| Função de Ativação        | ReLU                           |
| `dropout`                 | 0,1                            |
| Função de Perda (*loss*)  | *Binary Cross-Entropy*         |
| Otimização                | Adam (`lr = 0.001`)            |
| `batch_size`              | 1024                           |
| Épocas                    | 50                             |
| Entrada                   | (`user_id`, `item_id`)         |

---

### Técnica 4 – RAG + LLM

| Configuração                     | Detalhe                                   |
|----------------------------------|-------------------------------------------|
| Vector Store                     | ChromaDB                                  |
| Embeddings                       | *all-mpnet-base-v2* (dim = 768)          |
| `top-k` padrão                   | 10                                        |
| LLM (Modelo de Linguagem)        | `gemini-2.5-flash-lite`                  |
| Temperatura                      | 0,3                                      |
| `max_tokens`                     | 512                                      |
| Pré-processamento da *query*     | Expandida pelo LLM                       |
| Saída                            | Recomendações + explicação textual       |

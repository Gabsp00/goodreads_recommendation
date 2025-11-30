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

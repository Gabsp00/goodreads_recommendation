from pathlib import Path

# Paths do projeto
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BOOKS_PATH = PROCESSED_DATA_DIR / "books.csv"
USERS_PATH = PROCESSED_DATA_DIR / "users.csv" 
INTERACTIONS_TRAIN_PATH = PROCESSED_DATA_DIR / "interactions_train.csv"
INTERACTIONS_VAL_PATH = PROCESSED_DATA_DIR / "interactions_val.csv"
INTERACTIONS_TEST_PATH = PROCESSED_DATA_DIR / "interactions_test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Reprodutibilidade
RANDOM_SEED = 42

# Filtro do dataset (UCSD Goodreads)
MIN_USER_INTERACTIONS = 20
MIN_ITEM_INTERACTIONS = 10
TARGET_NUM_USERS = 40_000       # usuários finais
TARGET_NUM_ITEMS = 30_000       # livros finais

# Faixas de interações por usuário para val e test (últimas/penúltimas)
VAL_INTERACTIONS_RANGE = (2, 3)    # penúltimas interações -> val
TEST_INTERACTIONS_RANGE = (2, 3)   # últimas interações -> test

# Texto & TF-IDF (Técnica 1)
TEXT_FIELDS = ("title", "description")
MAX_TOKENS_TEXT = 512
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 40_000 

# Embeddings semânticos (Técnica 2)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 128

# Técnica 3 – NeuralCF
NEURAL_EMBED_DIM = 64
NEURAL_HIDDEN_DIMS = (128, 64)
NEURAL_DROPOUT = 0.2
NEURAL_LR = 1e-3
NEURAL_BATCH_SIZE = 1024
NEURAL_NUM_EPOCHS = 15
NEURAL_USE_GPU = True

# Vector DB / RAG (Técnica 4)
VECTOR_DB_IMPLEMENTATION = "chroma"
VECTOR_COLLECTION_NAME = "books_collection"

# LLM (Técnica 4)
LLM_PROVIDER = "gemini"
LLM_MODEL_NAME = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 512

OPENAI_API_KEY_ENVVAR = "OPENAI-KEY"
GEMINI_API_KEY_ENVVAR = "GEMINI_API_KEY"

MIN_RATING_FOR_POSITIVE = 4

# Config de recomendação
TOP_K_VALUES = (5, 10, 50, 200)
METRICS = ("precision", "recall", "ndcg")
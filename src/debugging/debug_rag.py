from config import INTERACTIONS_TEST_PATH
import pandas as pd
from src.evaluation.evaluate_models import _build_relevant_items_by_user
from src.evaluation.evaluate_models import evaluate_rag_retriever
from src.rag.vector_store import query_vector_store
from src.evaluation.evaluate_models import _build_user_profiles_from_train
from config import INTERACTIONS_TRAIN_PATH, BOOKS_PATH

train_df = pd.read_csv(INTERACTIONS_TRAIN_PATH)
test_df = pd.read_csv(INTERACTIONS_TEST_PATH)
books_df = pd.read_csv(BOOKS_PATH)

relevant_by_user = _build_relevant_items_by_user(test_df)
user_profiles = _build_user_profiles_from_train(train_df, books_df)

# pega um usuário com perfil e teste
u = next(iter(set(user_profiles.keys()) & set(relevant_by_user.keys())))

print("User:", u)
print("Relevantes no teste:", relevant_by_user[u])

query_text = user_profiles[u]
retrieved = query_vector_store(query_text, k=20)
recs = [int(r["book_id"]) for r in retrieved]

print("IDs recuperados:", recs)
print("Interseção:", set(recs) & relevant_by_user[u])
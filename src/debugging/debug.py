import pandas as pd

from config import INTERACTIONS_TEST_PATH
from src.evaluation.evaluate_models import _build_relevant_items_by_user
from src.models.neural_cf import generate_recommendations_for_all_users

# 1) Carrega teste e constrói relevantes
test_df = pd.read_csv(INTERACTIONS_TEST_PATH)
relevant_by_user = _build_relevant_items_by_user(test_df)

# 2) Gera recomendações do Neural CF
recs_by_user = generate_recommendations_for_all_users(k=10)

user_id = "0004a0bcdd96ce79bfccae1ff9459383"

print("Relevantes desse usuário no teste:")
print(relevant_by_user.get(user_id))

print("\nRecomendações do modelo para esse usuário:")
print(recs_by_user.get(user_id))

print("\nInterseção (itens que são relevantes E recomendados):")
print(set(recs_by_user.get(user_id, [])) & relevant_by_user.get(user_id, set()))
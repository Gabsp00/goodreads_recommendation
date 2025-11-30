"""
Implementações das métricas de avaliação para top-N recomendação:

- precision_at_k
- recall_at_k
- ndcg_at_k

Essas funções devem ser usadas por TODAS as técnicas (1 a 4).
"""

from typing import Sequence
import math

def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """
    Precision@k: (# itens recomendados entre os relevantes) / k
    """
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k

def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """
    Recall@k: (# itens relevantes recuperados) / (# total de relevantes)
    """
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)

def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """
    NDCG@k com relevância binária (1 se item é relevante, 0 caso contrário).
    """
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k, start=1):
        rel_i = 1.0 if item in relevant else 0.0
        if rel_i > 0:
            dcg += rel_i / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
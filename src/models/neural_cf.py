"""
Técnica 3: Collaborative Filtering neural (por exemplo, NeuMF ou
matrix factorization com embeddings de usuários e itens).

Este arquivo será a base principal da parte 3 do projeto (do outro membro da dupla).

Responsabilidades esperadas:

- Carregar interactions_train/val/test de data/processed.
- Criar um modelo em PyTorch com:
  - Embeddings de usuários
  - Embeddings de itens (livros)
  - Camadas adicionais (MLP) se desejado
- Treinar o modelo para prever ratings ou probabilidade de interação.
- Gerar recomendações top-k por usuário para avaliação (Precision@k, etc.).
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    INTERACTIONS_TRAIN_PATH,
    INTERACTIONS_VAL_PATH,
    INTERACTIONS_TEST_PATH,
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    BOOKS_PATH,
)

@dataclass
class NeuralCFConfig:
    """
    Configurações específicas para o modelo neural de CF.
    """
    embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 1024
    num_epochs: int = 50
    use_gpu: bool = True

class NeuralCFModel(nn.Module):
    """
    Modelo simples de CF neural (user/item embeddings + MLP).
    """
    def __init__(self, num_users: int, num_items: int, config: NeuralCFConfig):
        super().__init__()
        self.config = config

        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, config.embedding_dim)

        layers = []
        input_dim = 2 * config.embedding_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_idx)
        i = self.item_embedding(item_idx)
        x = torch.cat([u, i], dim=-1)
        logit = self.mlp(x).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob

class InteractionsDataset(Dataset):
    """
    Dataset de pares (user_idx, item_idx, label) para treino/val.
    """
    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        labels: np.ndarray,
    ):
        assert len(user_indices) == len(item_indices) == len(labels)
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.labels[idx],
        )

def _build_id_mappings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Cria mapeamentos user_id/book_id -> índices inteiros.
    Usa apenas train (+val opcional) para garantir consistência.
    """
    print("build_id_mappings...")
    user_ids = set(train_df["user_id"].unique())
    item_ids = set(train_df["book_id"].unique())

    if val_df is not None:
        user_ids.update(val_df["user_id"].unique())
        item_ids.update(val_df["book_id"].unique())

    user_ids = sorted(user_ids)
    item_ids = sorted(item_ids)

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {b: i for i, b in enumerate(item_ids)}

    return user2idx, item2idx

def _add_negative_samples(
    df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    negatives_per_positive: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera amostras negativas para feedback implícito.
    """
    print("adding negative samples...")
    rng = np.random.default_rng(RANDOM_SEED)

    pos_users = df["user_id"].map(user2idx).to_numpy()
    pos_items = df["book_id"].map(item2idx).to_numpy()
    pos_labels = np.ones(len(df), dtype=np.float32)

    user_pos_items: Dict[int, set] = (
        df.groupby("user_id")["book_id"].apply(set).to_dict()
    )

    all_item_ids = np.array(list(item2idx.keys()))
    neg_users = []
    neg_items = []

    for user_id, pos_items_set in user_pos_items.items():
        u_idx = user2idx[user_id]
        pos_count = len(pos_items_set)
        num_neg = max(1, pos_count * negatives_per_positive)

        sampled = 0
        while sampled < num_neg:
            candidate_item_id = rng.choice(all_item_ids)
            if candidate_item_id not in pos_items_set:
                neg_users.append(u_idx)
                neg_items.append(item2idx[candidate_item_id])
                sampled += 1

    neg_labels = np.zeros(len(neg_users), dtype=np.float32)

    users = np.concatenate([pos_users, np.array(neg_users, dtype=np.int64)])
    items = np.concatenate([pos_items, np.array(neg_items, dtype=np.int64)])
    labels = np.concatenate([pos_labels, neg_labels])

    return users, items, labels

def train_neural_cf(config: NeuralCFConfig) -> Dict[str, Any]:
    print("training neural_cf")

    """
    Função principal de treinamento do modelo de CF neural.

    Faz:
    - Carrega dados de treino/val.
    - Cria mapeamentos user_id/book_id -> índices.
    - Gera amostras negativas.
    - Treina o modelo.
    - Salva pesos em models/neural_cf.pt.
    - Retorna métricas de loss médio em train/val.
    """
    print("training model...")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
    )

    train_df = pd.read_csv(INTERACTIONS_TRAIN_PATH)
    val_df = pd.read_csv(INTERACTIONS_VAL_PATH)

    user2idx, item2idx = _build_id_mappings(train_df, val_df)
    num_users = len(user2idx)
    num_items = len(item2idx)

    train_users, train_items, train_labels = _add_negative_samples(
        train_df, user2idx, item2idx, negatives_per_positive=1
    )
    val_users, val_items, val_labels = _add_negative_samples(
        val_df, user2idx, item2idx, negatives_per_positive=1
    )

    train_dataset = InteractionsDataset(train_users, train_items, train_labels)
    val_dataset = InteractionsDataset(val_users, val_items, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = NeuralCFModel(num_users, num_items, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    criterion = nn.BCELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
    }
    print("starting...")
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            user_idx, item_idx, labels = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(user_idx, item_idx)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                user_idx, item_idx, labels = batch
                user_idx = user_idx.to(device)
                item_idx = item_idx.to(device)
                labels = labels.to(device)

                preds = model(user_idx, item_idx)
                loss = criterion(preds, labels)
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

        print(
            f"[Epoch {epoch}/{config.num_epochs}] "
            f"train_loss={mean_train:.4f} val_loss={mean_val:.4f}"
        )

    models_dir = PROCESSED_DATA_DIR.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "neural_cf.pt"
    print("Training finished. Saving model...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
            "config": config.__dict__,
        },
        model_path,
    )
    print(f"Modelo salvo em {model_path}")

    return {
        "history": history,
        "num_users": num_users,
        "num_items": num_items,
    }

def generate_recommendations_for_all_users(
    k: int = 200,
    use_gpu: bool = True,
) -> Dict[int, list[int]]:
    """
    Gera recomendações top-k para todos os usuários.

    Retorna:
        dict[user_id -> lista de book_id recomendados]
    """
    print("generate_recommendations_for_all_users ...")
    print(f"gerando recomendações: k={k}")

    df_train = pd.read_csv(INTERACTIONS_TRAIN_PATH, usecols=["user_id", "book_id"])
    print(df_train.head())
    print(df_train.columns)

    user_consumed: Dict[int, set[int]] = (
        df_train.groupby("user_id")["book_id"].apply(set).to_dict()
    )

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    model_path = PROCESSED_DATA_DIR.parent / "models" / "neural_cf.pt"
    print(f"carregando pesos de {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    saved_config = checkpoint.get("config", {})
    config = (
        NeuralCFConfig(**saved_config)
        if isinstance(saved_config, dict)
        else NeuralCFConfig()
    )

    user2idx: Dict[int, int] = checkpoint["user2idx"]
    item2idx: Dict[int, int] = checkpoint["item2idx"]

    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"num_users={num_users}, num_items={num_items}")

    model = NeuralCFModel(num_users=num_users, num_items=num_items, config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_item_id: Dict[int, int] = {
        idx: book_id for book_id, idx in item2idx.items()
    }

    all_item_idxs = torch.arange(num_items, device=device)

    recs_by_user: Dict[int, List[int]] = {}

    print("generating recommendations...")
    with torch.no_grad():
        for user_id in tqdm(sorted(user2idx.keys()), desc="users"):
            u_idx = user2idx[user_id]

            user_idxs = torch.full(
                (num_items,), u_idx, dtype=torch.long, device=device
            )

            scores = model(user_idxs, all_item_idxs).cpu().numpy()

            consumed_books = user_consumed.get(user_id, set())
            consumed_item_idxs = [
                item2idx[b] for b in consumed_books if b in item2idx
            ]

            if consumed_item_idxs:
                scores[consumed_item_idxs] = -1e9

            if k >= num_items:
                topk_item_idxs = np.argsort(-scores)
            else:
                topk_part = np.argpartition(-scores, k - 1)[:k]
                topk_item_idxs = topk_part[np.argsort(-scores[topk_part])]

            rec_books = [idx_to_item_id[i] for i in topk_item_idxs]
            recs_by_user[user_id] = rec_books

    return recs_by_user

def _load_trained_neural_cf_model(
    use_gpu: bool = True,
) -> Tuple[NeuralCFModel, Dict[int, int], Dict[int, int], torch.device]:
    """
    Carrega do disco o modelo NeuralCF treinado, juntamente com os mapeamentos.
    Reutilizado tanto para avaliação offline quanto para uso em produção/Streamlit.
    """
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    model_path = PROCESSED_DATA_DIR.parent / "models" / "neural_cf.pt"
    print(f"carregando pesos de {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    saved_config = checkpoint.get("config", {})
    config = (
        NeuralCFConfig(**saved_config)
        if isinstance(saved_config, dict)
        else NeuralCFConfig()
    )

    user2idx: Dict[int, int] = checkpoint["user2idx"]
    item2idx: Dict[int, int] = checkpoint["item2idx"]

    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"num_users={num_users}, num_items={num_items}")

    model = NeuralCFModel(num_users=num_users, num_items=num_items, config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, user2idx, item2idx, device

def recommend_for_user_neural_cf(
    user_id: int,
    k: int = 10,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Gera recomendações top-k para UM usuário específico, retornando um DataFrame
    no mesmo padrão das outras técnicas (colunas: book_id, title, score).

    - Usa o modelo Neural CF treinado.
    - Remove livros que o usuário já consumiu.
    """
    df_train = pd.read_csv(INTERACTIONS_TRAIN_PATH, usecols=["user_id", "book_id"])
    user_consumed: Dict[int, set[int]] = (
        df_train.groupby("user_id")["book_id"].apply(set).to_dict()
    )

    model, user2idx, item2idx, device = _load_trained_neural_cf_model(
        use_gpu=use_gpu
    )

    if user_id not in user2idx:
        raise ValueError(
            f"user_id {user_id} não está presente no modelo treinado de CF neural."
        )

    num_items = len(item2idx)
    all_item_idxs = torch.arange(num_items, device=device)

    u_idx = user2idx[user_id]
    user_idxs = torch.full(
        (num_items,), u_idx, dtype=torch.long, device=device
    )

    with torch.no_grad():
        scores = model(user_idxs, all_item_idxs).cpu().numpy()

    consumed_books = user_consumed.get(user_id, set())
    consumed_item_idxs = [item2idx[b] for b in consumed_books if b in item2idx]
    if consumed_item_idxs:
        scores[consumed_item_idxs] = -1e9

    if k >= num_items:
        topk_item_idxs = np.argsort(-scores)
    else:
        topk_part = np.argpartition(-scores, k - 1)[:k]
        topk_item_idxs = topk_part[np.argsort(-scores[topk_part])]

    idx_to_item_id: Dict[int, int] = {idx: bid for bid, idx in item2idx.items()}

    top_book_ids = [idx_to_item_id[i] for i in topk_item_idxs]
    top_scores = [float(scores[i]) for i in topk_item_idxs]

    books_df = pd.read_csv(BOOKS_PATH).set_index("book_id")

    rows = []
    for bid, s in zip(top_book_ids, top_scores):
        if bid in books_df.index:
            row = books_df.loc[bid].to_dict()
        else:
            row = {"title": None, "description": None, "text": None}
        row["book_id"] = bid
        row["score"] = s
        rows.append(row)

    recs_df = pd.DataFrame(rows)

    cols = [c for c in ["book_id", "title", "score"] if c in recs_df.columns]
    return recs_df[cols]

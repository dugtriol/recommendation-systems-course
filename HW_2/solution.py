import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking


def train_bpr_and_recommend(train: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    unique_users = train["user_id"].unique()
    unique_items = train["item_id"].unique()

    n_users = len(unique_users)
    n_items = len(unique_items)

    user_to_index = {uid: i for i, uid in enumerate(unique_users)}
    item_to_index = {iid: i for i, iid in enumerate(unique_items)}

    index_to_user = np.array(unique_users)
    index_to_item = np.array(unique_items)

    rows = train["user_id"].map(user_to_index).to_numpy()
    cols = train["item_id"].map(item_to_index).to_numpy()

    data = np.ones(len(rows), dtype="float32")
    user_item_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items),
        dtype="float32",
    )

    model = BayesianPersonalizedRanking(
        factors=64,
        learning_rate=0.05,
        regularization=0.01,
        iterations=50,
        random_state=42,
    )
    model.fit(user_item_matrix)

    user_factors = model.user_factors
    item_factors = model.item_factors
    scores = user_factors @ item_factors.T

    recommendations = []

    indptr = user_item_matrix.indptr
    indices = user_item_matrix.indices

    for u_idx in range(n_users):
        user_id = index_to_user[u_idx]
        user_scores = scores[u_idx].copy()

        start = indptr[u_idx]
        end = indptr[u_idx + 1]
        seen_items = indices[start:end]
        user_scores[seen_items] = -np.inf

        if np.all(~np.isfinite(user_scores)):
            continue

        if top_n < n_items:
            top_idx = np.argpartition(-user_scores, top_n)[:top_n]
        else:
            top_idx = np.argsort(-user_scores)

        top_idx = top_idx[np.argsort(-user_scores[top_idx])]

        for item_idx in top_idx:
            item_id = index_to_item[item_idx]
            recommendations.append({"user_id": user_id, "recs": item_id})

    result = pd.DataFrame(recommendations)
    return result


def main(
    input_path_interactions: str,
    input_path_users: str,
    input_path_items: str,
    input_path_embeddings: str,
    output_path: str,
):
    train = pd.read_parquet(input_path_interactions)

    _ = input_path_users
    _ = input_path_items
    _ = input_path_embeddings

    recs_df = train_bpr_and_recommend(train, top_n=10)
    recs_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_interactions",
        type=str,
        required=True,
        help="Path to train interactions parquet file",
    )
    parser.add_argument(
        "--input_path_users",
        type=str,
        required=True,
        help="Path to user_metadata parquet file (not used in this solution)",
    )
    parser.add_argument(
        "--input_path_items",
        type=str,
        required=True,
        help="Path to item_metadata parquet file (not used in this solution)",
    )
    parser.add_argument(
        "--input_path_embeddings",
        type=str,
        required=True,
        help="Path to item_embeddings parquet file (not used in this solution)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output CSV with recommendations (user_id,recs)",
    )
    args = parser.parse_args()

    main(
        args.input_path_interactions,
        args.input_path_users,
        args.input_path_items,
        args.input_path_embeddings,
        args.output_path,
    )

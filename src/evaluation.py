import numpy as np
import pandas as pd

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))

def average_precision_at_k(recommended_items, relevant_items, k=10):
    """Compute Average Precision @ K for a single user.
    
    Args:
        recommended_items (list): Ranked list of recommended item IDs.
        relevant_items (set): Set of relevant item IDs in the test set.
        k (int): Cut-off threshold.
        
    Returns:
        float: Average Precision @ K.
    """
    if not relevant_items:
        return 0.0
        
    recommended_items = list(recommended_items)[:k]
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i
            
    # Denominator is min(len(relevant_items), k) to avoid penalizing users
    # who have fewer than k relevant items in the test set.
    return score / min(len(relevant_items), k)

def ndcg_at_k(recommended_items, relevant_items, k=10):
    """Compute Normalized Discounted Cumulative Gain @ K for a single user.
    
    Args:
        recommended_items (list): Ranked list of recommended item IDs.
        relevant_items (set): Set of relevant item IDs in the test set.
        k (int): Cut-off threshold.
        
    Returns:
        float: NDCG @ K.
    """
    if not relevant_items:
        return 0.0
        
    recommended_items = list(recommended_items)[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2.0)
            
    # IDCG is the DCG of a perfect recommendation list
    idcg = 0.0
    ideal_hits = min(len(relevant_items), k)
    for i in range(ideal_hits):
        idcg += 1.0 / np.log2(i + 2.0)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def evaluate_predictions(test_df, model):
    """Evaluate rating prediction accuracy (RMSE and MAE)."""
    y_true = test_df['rating'].values
    y_pred = []
    
    for row in test_df.itertuples():
        pred = model.predict(row.user_id, row.movie_id)
        y_pred.append(pred)
        
    y_pred = np.array(y_pred)
    
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae
    }

def evaluate_ranking(train_df, test_df, model, k=10, relevance_threshold=3.5, sample_users_pct=0.1, random_state=42):
    """Evaluate recommendation ranking performance (MAP@10, Precision@10, Recall@10, NDCG@10).
    
    Because generating Top-10 recommendations for all 10,000 users across 2,000 movies
    can be computationally heavy, we evaluate ranking on a representative sample of users
    (e.g., 10% of users, default 1,000 users) if sample_users_pct is set, or evaluate on all users.
    
    Args:
        train_df (pd.DataFrame): Training ratings.
        test_df (pd.DataFrame): Test ratings.
        model (BaseRecommender): Model to evaluate.
        k (int): Number of recommendations to evaluate.
        relevance_threshold (float): Threshold to count a rating as relevant.
        sample_users_pct (float): Percentage of test users to evaluate. If 1.0, evaluates all.
        random_state (int): Seed for sampling users.
    """
    # 1. Identify relevant items for each user in the test set
    # User -> set(relevant movie_ids)
    relevant_test_ratings = test_df[test_df['rating'] >= relevance_threshold]
    user_to_relevant_items = relevant_test_ratings.groupby('user_id')['movie_id'].apply(set).to_dict()
    
    # 2. Get list of users who have at least one relevant rating in the test set
    eval_users = list(user_to_relevant_items.keys())
    
    if sample_users_pct < 1.0:
        np.random.seed(random_state)
        sample_size = max(1, int(len(eval_users) * sample_users_pct))
        eval_users = np.random.choice(eval_users, size=sample_size, replace=False)
        print(f"Evaluating ranking performance on a sample of {len(eval_users)} users (out of {len(user_to_relevant_items)} users with relevant ratings)...")
    else:
        print(f"Evaluating ranking performance on all {len(eval_users)} users with relevant ratings...")
        
    ap_scores = []
    precisions = []
    recalls = []
    ndcgs = []
    
    for u_id in eval_users:
        relevant_items = user_to_relevant_items[u_id]
        
        # Generate Top-k recommendations
        # Exclude items the user rated in the training set
        recommendations_with_scores = model.recommend(u_id, n_items=k, exclude_rated_items=True)
        recommended_items = [item_id for item_id, _ in recommendations_with_scores]
        
        # Compute metrics
        ap = average_precision_at_k(recommended_items, relevant_items, k=k)
        ndcg = ndcg_at_k(recommended_items, relevant_items, k=k)
        
        # Precision@K: number of relevant recommended items / K
        hits = len(set(recommended_items) & relevant_items)
        precision = hits / k
        
        # Recall@K: number of relevant recommended items / total relevant items in test
        recall = hits / len(relevant_items)
        
        ap_scores.append(ap)
        ndcgs.append(ndcg)
        precisions.append(precision)
        recalls.append(recall)
        
    return {
        f'map_at_{k}': float(np.mean(ap_scores)),
        f'ndcg_at_{k}': float(np.mean(ndcgs)),
        f'precision_at_{k}': float(np.mean(precisions)),
        f'recall_at_{k}': float(np.mean(recalls))
    }

def print_evaluation_summary(pred_metrics, rank_metrics, model_name="Model"):
    print(f"\n==================== {model_name} Evaluation ====================")
    print(f"Rating Prediction Metrics:")
    print(f"  RMSE: {pred_metrics['rmse']:.4f}")
    print(f"  MAE:  {pred_metrics['mae']:.4f}")
    print(f"\nRecommendation Ranking Metrics (@10):")
    print(f"  MAP@10:       {rank_metrics['map_at_10']:.4f}")
    print(f"  NDCG@10:      {rank_metrics['ndcg_at_10']:.4f}")
    print(f"  Precision@10: {rank_metrics['precision_at_10']:.4f}")
    print(f"  Recall@10:    {rank_metrics['recall_at_10']:.4f}")
    print("=========================================================\n")

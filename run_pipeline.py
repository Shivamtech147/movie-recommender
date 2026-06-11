import os
import time
import pickle
import json
import pandas as pd
import numpy as np
import torch
from src.data_processing import run_pipeline as run_data_prep
from src.eda import run_eda
from src.models.collaborative_filtering import ItemBasedCF
from src.models.matrix_factorization import MatrixFactorizationRecommender
from src.models.ncf import NeuralCollabFiltering
from src.evaluation import evaluate_predictions, evaluate_ranking, print_evaluation_summary

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    start_time = time.time()
    ensure_directories()
    
    print("==================================================================")
    print("   STARTING END-TO-END RECOMMENDATION SYSTEM WORKFLOW")
    print("==================================================================")
    
    # 1. Run Data Preparation (will skip download/extraction if already done)
    print("\n--- STEP 1: Data Preparation ---")
    run_data_prep()
    
    # 2. Run Exploratory Data Analysis
    print("\n--- STEP 2: Exploratory Data Analysis ---")
    run_eda()
    
    # Load processed data
    print("\nLoading train/test datasets...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    movie_titles_df = pd.read_csv(os.path.join(PROCESSED_DIR, "movie_titles.csv"))
    
    # Initialize comparison results dictionary
    results = {}
    
    # --------------------------------------------------------
    # Model 1: Item-Based Collaborative Filtering (IBCF)
    # --------------------------------------------------------
    print("\n--- STEP 3: Training Model 1 - Item-Based Collaborative Filtering ---")
    model_cf = ItemBasedCF(k_neighbors=40, shrinkage=10)
    
    t_start = time.time()
    model_cf.fit(train_df)
    train_time_cf = time.time() - t_start
    
    print("Evaluating Item-Based CF...")
    t_start = time.time()
    cf_pred_metrics = evaluate_predictions(test_df, model_cf)
    cf_rank_metrics = evaluate_ranking(train_df, test_df, model_cf, k=10, relevance_threshold=3.5, sample_users_pct=0.1)
    eval_time_cf = time.time() - t_start
    
    print_evaluation_summary(cf_pred_metrics, cf_rank_metrics, "Item-Based CF")
    
    results['Item-Based CF'] = {
        'Train Time (s)': train_time_cf,
        'Eval Time (s)': eval_time_cf,
        'RMSE': cf_pred_metrics['rmse'],
        'MAE': cf_pred_metrics['mae'],
        'MAP@10': cf_rank_metrics['map_at_10'],
        'NDCG@10': cf_rank_metrics['ndcg_at_10'],
        'Precision@10': cf_rank_metrics['precision_at_10'],
        'Recall@10': cf_rank_metrics['recall_at_10']
    }
    
    # Save CF Model
    cf_model_path = os.path.join(MODELS_DIR, "item_cf.pkl")
    with open(cf_model_path, 'wb') as f:
        pickle.dump(model_cf, f)
    print(f"Saved Item-Based CF model to {cf_model_path}")
    
    # --------------------------------------------------------
    # Model 2: Matrix Factorization (PyTorch)
    # --------------------------------------------------------
    print("\n--- STEP 4: Training Model 2 - PyTorch Matrix Factorization with Biases ---")
    model_mf = MatrixFactorizationRecommender(embedding_dim=50, lr=0.005, weight_decay=0.01, epochs=12, batch_size=512)
    
    t_start = time.time()
    model_mf.fit(train_df)
    train_time_mf = time.time() - t_start
    
    print("Evaluating Matrix Factorization...")
    t_start = time.time()
    mf_pred_metrics = evaluate_predictions(test_df, model_mf)
    mf_rank_metrics = evaluate_ranking(train_df, test_df, model_mf, k=10, relevance_threshold=3.5, sample_users_pct=0.1)
    eval_time_mf = time.time() - t_start
    
    print_evaluation_summary(mf_pred_metrics, mf_rank_metrics, "Matrix Factorization")
    
    results['Matrix Factorization'] = {
        'Train Time (s)': train_time_mf,
        'Eval Time (s)': eval_time_mf,
        'RMSE': mf_pred_metrics['rmse'],
        'MAE': mf_pred_metrics['mae'],
        'MAP@10': mf_rank_metrics['map_at_10'],
        'NDCG@10': mf_rank_metrics['ndcg_at_10'],
        'Precision@10': mf_rank_metrics['precision_at_10'],
        'Recall@10': mf_rank_metrics['recall_at_10']
    }
    
    # Save MF Model
    # Move model to CPU before pickling to prevent device loading errors on other machines
    model_mf.model.cpu()
    model_mf.device = torch.device('cpu')
    mf_model_path = os.path.join(MODELS_DIR, "matrix_factorization.pkl")
    with open(mf_model_path, 'wb') as f:
        pickle.dump(model_mf, f)
    print(f"Saved Matrix Factorization model to {mf_model_path}")
    
    # --------------------------------------------------------
    # Model 3: Neural Collaborative Filtering (PyTorch NCF)
    # --------------------------------------------------------
    print("\n--- STEP 5: Training Model 3 - PyTorch Neural Collaborative Filtering ---")
    model_ncf = NeuralCollabFiltering(latent_dim_gmf=32, latent_dim_mlp=32, layers=[64, 32, 16, 8],
                                      lr=0.001, weight_decay=1e-5, epochs=10, batch_size=512)
    
    t_start = time.time()
    model_ncf.fit(train_df)
    train_time_ncf = time.time() - t_start
    
    print("Evaluating Neural Collaborative Filtering...")
    t_start = time.time()
    ncf_pred_metrics = evaluate_predictions(test_df, model_ncf)
    ncf_rank_metrics = evaluate_ranking(train_df, test_df, model_ncf, k=10, relevance_threshold=3.5, sample_users_pct=0.1)
    eval_time_ncf = time.time() - t_start
    
    print_evaluation_summary(ncf_pred_metrics, ncf_rank_metrics, "Neural CF (NCF)")
    
    results['Neural CF (NCF)'] = {
        'Train Time (s)': train_time_ncf,
        'Eval Time (s)': eval_time_ncf,
        'RMSE': ncf_pred_metrics['rmse'],
        'MAE': ncf_pred_metrics['mae'],
        'MAP@10': ncf_rank_metrics['map_at_10'],
        'NDCG@10': ncf_rank_metrics['ndcg_at_10'],
        'Precision@10': ncf_rank_metrics['precision_at_10'],
        'Recall@10': ncf_rank_metrics['recall_at_10']
    }
    
    # Save NCF Model
    model_ncf.model.cpu()
    model_ncf.device = torch.device('cpu')
    ncf_model_path = os.path.join(MODELS_DIR, "ncf.pkl")
    with open(ncf_model_path, 'wb') as f:
        pickle.dump(model_ncf, f)
    print(f"Saved NCF model to {ncf_model_path}")
    
    # --------------------------------------------------------
    # Model Comparison & Reporting
    # --------------------------------------------------------
    print("\n--- STEP 6: Model Comparison & Reporting ---")
    df_comparison = pd.DataFrame(results).T
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df_comparison.to_csv(comparison_path)
    
    print("\n========================= MODEL COMPARISON =========================")
    print(df_comparison.to_string())
    print("====================================================================\n")
    print(f"Saved model comparison results to {comparison_path}")
    
    elapsed_time = time.time() - start_time
    print(f"Entire recommendation system workflow finished in {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()

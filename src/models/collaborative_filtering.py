import numpy as np
import pandas as pd
from src.models.base_model import BaseRecommender

class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering using Cosine Similarity."""
    
    def __init__(self, k_neighbors=40, shrinkage=10):
        """
        Args:
            k_neighbors (int): Number of similar items to consider for prediction.
            shrinkage (int): Regularization factor for similarity calculation to handle low co-ratings.
        """
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        
        # Internal states
        self.global_mean = 3.5
        self.user_means = {}
        self.movie_means = {}
        self.similarity_matrix = None
        self.movie_id_to_idx = {}
        self.movie_idx_to_id = {}
        self.user_id_to_idx = {}
        self.user_idx_to_id = {}
        
        # Dense matrices
        self.train_ratings = None  # U x M dense matrix
        self.rated_mask = None     # U x M boolean mask
        self.predictions = None    # U x M precomputed predictions
        
    def fit(self, train_df):
        print("Fitting Item-Based Collaborative Filtering model...")
        
        # Calculate averages for fallbacks
        self.global_mean = train_df['rating'].mean()
        self.user_means = train_df.groupby('user_id')['rating'].mean().to_dict()
        self.movie_means = train_df.groupby('movie_id')['rating'].mean().to_dict()
        
        # Map user and movie IDs to sequential indices
        unique_movies = sorted(train_df['movie_id'].unique())
        unique_users = sorted(train_df['user_id'].unique())
        
        self.movie_id_to_idx = {m_id: idx for idx, m_id in enumerate(unique_movies)}
        self.movie_idx_to_id = {idx: m_id for idx, m_id in enumerate(unique_movies)}
        self.user_id_to_idx = {u_id: idx for idx, u_id in enumerate(unique_users)}
        self.user_idx_to_id = {idx: u_id for idx, u_id in enumerate(unique_users)}
        
        num_users = len(unique_users)
        num_movies = len(unique_movies)
        
        # Create dense rating matrix
        self.train_ratings = np.zeros((num_users, num_movies), dtype=np.float32)
        for row in train_df.itertuples():
            u_idx = self.user_id_to_idx[row.user_id]
            m_idx = self.movie_id_to_idx[row.movie_id]
            self.train_ratings[u_idx, m_idx] = row.rating
            
        self.rated_mask = (self.train_ratings > 0)
        
        # Center the rating matrix by subtracting user means (for Adjusted Cosine)
        user_means_vector = np.array([self.user_means[u_id] for u_id in unique_users], dtype=np.float32).reshape(-1, 1)
        centered_ratings = np.zeros_like(self.train_ratings)
        centered_ratings[self.rated_mask] = self.train_ratings[self.rated_mask] - user_means_vector[np.where(self.rated_mask)[0], 0]
        
        print("Calculating movie similarity matrix (Adjusted Cosine)...")
        # Compute cosine similarity between movie columns of centered matrix
        # sim(i, j) = dot(centered[:, i], centered[:, j]) / (norm(centered[:, i]) * norm(centered[:, j]))
        dot_product = np.dot(centered_ratings.T, centered_ratings)
        
        # Norms of columns
        norms = np.linalg.norm(centered_ratings, axis=0)
        norms_outer = np.outer(norms, norms)
        
        # Calculate similarity with shrinkage to penalize items with few overlapping ratings
        # sim = dot_product / (norms_outer + shrinkage)
        self.similarity_matrix = np.zeros_like(dot_product)
        non_zero_mask = (norms_outer > 0)
        self.similarity_matrix[non_zero_mask] = dot_product[non_zero_mask] / (norms_outer[non_zero_mask] + self.shrinkage)
        
        # Set self-similarity to 0 to avoid recommending a movie because of itself
        np.fill_diagonal(self.similarity_matrix, 0.0)
        
        # Clean up negative similarities (usually we only keep positive relationships)
        self.similarity_matrix = np.clip(self.similarity_matrix, a_min=0, a_max=1.0)
        
        print("Pre-computing predictions for all users and items...")
        # For item-based CF: Pred(u, i) = user_mean(u) + sum(sim(i, j) * centered_rating(u, j)) / sum(|sim(i, j)|)
        # We limit to top-K neighbors:
        sim_top_k = self.similarity_matrix.copy()
        
        # For each movie, keep only the top k similarities, set others to 0
        for i in range(num_movies):
            row = sim_top_k[i]
            if len(row) > self.k_neighbors:
                top_k_indices = np.argpartition(row, -self.k_neighbors)[-self.k_neighbors:]
                mask = np.ones(len(row), dtype=bool)
                mask[top_k_indices] = False
                row[mask] = 0.0
                
        # Compute numerator: centered_ratings * sim_top_k
        # Shape: (U x M) * (M x M) -> U x M
        numerator = np.dot(centered_ratings, sim_top_k.T)
        
        # Compute denominator: rated_mask * sim_top_k
        # Shape: (U x M) * (M x M) -> U x M
        denominator = np.dot(self.rated_mask.astype(np.float32), sim_top_k.T)
        
        # Calculate predictions
        self.predictions = np.zeros_like(self.train_ratings)
        valid_mask = (denominator > 0)
        self.predictions[valid_mask] = user_means_vector[np.where(valid_mask)[0], 0] + (numerator[valid_mask] / denominator[valid_mask])
        
        # Fallback for users/items with no overlapping neighbors: user mean or movie mean or global mean
        fallback_mask = ~valid_mask
        for u_idx in range(num_users):
            u_id = self.user_idx_to_id[u_idx]
            u_mean = self.user_means.get(u_id, self.global_mean)
            self.predictions[u_idx, fallback_mask[u_idx]] = u_mean
            
        # Clip predictions to valid rating scale [1, 5]
        self.predictions = np.clip(self.predictions, 1.0, 5.0)
        print("Model fitting complete!")
        
    def predict(self, user_id, movie_id):
        # Fallback if user or movie is unseen (cold start)
        if user_id not in self.user_id_to_idx:
            return self.movie_means.get(movie_id, self.global_mean)
        if movie_id not in self.movie_id_to_idx:
            return self.user_means.get(user_id, self.global_mean)
            
        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        
        return float(self.predictions[u_idx, m_idx])
        
    def recommend(self, user_id, n_items=10, exclude_rated_items=True):
        if user_id not in self.user_id_to_idx:
            # Cold start user: recommend top popular/highly rated items
            # Return movie_id and global average rating
            sorted_movies = sorted(self.movie_means.items(), key=lambda x: x[1], reverse=True)
            return sorted_movies[:n_items]
            
        u_idx = self.user_id_to_idx[user_id]
        user_preds = self.predictions[u_idx].copy()
        
        if exclude_rated_items:
            # Set predictions for already rated items to a very low value
            user_preds[self.rated_mask[u_idx]] = -1.0
            
        # Get sorted movie indices
        top_indices = np.argsort(user_preds)[::-1][:n_items]
        
        recommendations = []
        for m_idx in top_indices:
            # Skip if it was excluded
            if user_preds[m_idx] < 0:
                continue
            movie_id = self.movie_idx_to_id[m_idx]
            recommendations.append((movie_id, float(user_preds[m_idx])))
            
        return recommendations

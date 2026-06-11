import os
import pandas as pd
import numpy as np

class RecommendationManager:
    """Manages recommendation generation, movie similarities, explainability, and cold start fallbacks."""
    
    def __init__(self, train_df, movie_titles_df):
        """
        Args:
            train_df (pd.DataFrame): The training dataset of ratings.
            movie_titles_df (pd.DataFrame): Movie metadata containing movie_id, year, title.
        """
        self.train_df = train_df
        self.movie_titles_df = movie_titles_df
        
        # Precompute movie stats for cold-start and fallbacks
        movie_stats = train_df.groupby('movie_id').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        )
        self.movie_stats = movie_stats.merge(movie_titles_df, on='movie_id', how='left')
        
        # Top popular movies (highest count of ratings)
        self.popular_movies = self.movie_stats.sort_values(by='rating_count', ascending=False)
        # Top highly-rated movies (min 100 ratings)
        self.highly_rated_movies = self.movie_stats[self.movie_stats['rating_count'] >= 100].sort_values(by='avg_rating', ascending=False)

    def get_movie_details(self, movie_id):
        """Get details for a given movie ID."""
        row = self.movie_titles_df[self.movie_titles_df['movie_id'] == movie_id]
        if not row.empty:
            return {
                'movie_id': movie_id,
                'title': row.iloc[0]['title'],
                'year': row.iloc[0]['year']
            }
        return {'movie_id': movie_id, 'title': 'Unknown Movie', 'year': np.nan}

    def get_user_history(self, user_id, n_items=10):
        """Get historical ratings for a given user in training set."""
        user_ratings = self.train_df[self.train_df['user_id'] == user_id].sort_values(by='rating', ascending=False)
        history = []
        for row in user_ratings.head(n_items).itertuples():
            details = self.get_movie_details(row.movie_id)
            history.append({
                'movie_id': row.movie_id,
                'title': details['title'],
                'year': details['year'],
                'rating': float(row.rating),
                'date': str(row.date)
            })
        return history

    def get_popular_recommendations(self, n_items=10):
        """Get popularity-based recommendations for cold-start users."""
        recs = []
        for row in self.popular_movies.head(n_items).itertuples():
            recs.append({
                'movie_id': int(row.movie_id),
                'title': row.title,
                'year': row.year,
                'predicted_rating': float(row.avg_rating),
                'score_type': 'popularity'
            })
        return recs

    def generate_personalized_recommendations(self, model, user_id, n_items=10, exclude_rated=True):
        """Generate personalized recommendations using a trained model.
        
        Falls back to popularity-based recommendations if the user is unseen (cold start).
        """
        # Check if user is in training set
        is_cold_start = False
        if hasattr(model, 'user_id_to_idx'):
            if user_id not in model.user_id_to_idx:
                is_cold_start = True
        else:
            # Fallback check
            if user_id not in self.train_df['user_id'].unique():
                is_cold_start = True
                
        if is_cold_start:
            print(f"User {user_id} not seen in training. Delivering popularity fallback recommendations.")
            return self.get_popular_recommendations(n_items)

        # Generate predictions from model
        model_recs = model.recommend(user_id, n_items=n_items, exclude_rated_items=exclude_rated)
        
        recs = []
        for movie_id, score in model_recs:
            details = self.get_movie_details(movie_id)
            recs.append({
                'movie_id': movie_id,
                'title': details['title'],
                'year': details['year'],
                'predicted_rating': float(score),
                'score_type': 'personalized'
            })
        return recs

    def get_similar_movies_item_cf(self, item_cf_model, movie_id, n_items=5):
        """Find similar movies using the cosine similarity matrix of Item-Based CF."""
        if movie_id not in item_cf_model.movie_id_to_idx:
            return []
            
        m_idx = item_cf_model.movie_id_to_idx[movie_id]
        sim_scores = item_cf_model.similarity_matrix[m_idx].copy()
        
        # Sort in descending order
        top_indices = np.argsort(sim_scores)[::-1][:n_items]
        
        similar_movies = []
        for idx in top_indices:
            sim_id = item_cf_model.movie_idx_to_id[idx]
            score = float(sim_scores[idx])
            # If similarity is 0, skip
            if score == 0:
                continue
            details = self.get_movie_details(sim_id)
            similar_movies.append({
                'movie_id': sim_id,
                'title': details['title'],
                'year': details['year'],
                'similarity': score
            })
        return similar_movies

    def get_similar_movies_latent(self, pytorch_model_wrapper, movie_id, n_items=5):
        """Find similar movies in the latent embedding space of a PyTorch model (MF or NCF)."""
        if movie_id not in pytorch_model_wrapper.movie_id_to_idx:
            return []
            
        m_idx = pytorch_model_wrapper.movie_id_to_idx[movie_id]
        model = pytorch_model_wrapper.model
        
        # Extract item embeddings
        if hasattr(model, 'movie_embeddings'):
            # MF model
            embeddings = model.movie_embeddings.weight.detach().cpu().numpy()
        elif hasattr(model, 'movie_embed_gmf') and hasattr(model, 'movie_embed_mlp'):
            # NCF model
            gmf_embed = model.movie_embed_gmf.weight.detach().cpu().numpy()
            mlp_embed = model.movie_embed_mlp.weight.detach().cpu().numpy()
            # Concatenate or choose one. Let's concatenate them!
            embeddings = np.concatenate([gmf_embed, mlp_embed], axis=1)
        else:
            return []
            
        # Target movie embedding
        target_embed = embeddings[m_idx]
        
        # Compute cosine similarity between target embedding and all embeddings
        norms = np.linalg.norm(embeddings, axis=1)
        target_norm = np.linalg.norm(target_embed)
        
        dot_product = np.dot(embeddings, target_embed)
        similarities = dot_product / (norms * target_norm + 1e-9)
        
        # Set target movie similarity to 0
        similarities[m_idx] = -1.0
        
        top_indices = np.argsort(similarities)[::-1][:n_items]
        
        similar_movies = []
        for idx in top_indices:
            sim_id = pytorch_model_wrapper.movie_idx_to_id[idx]
            score = float(similarities[idx])
            details = self.get_movie_details(sim_id)
            similar_movies.append({
                'movie_id': sim_id,
                'title': details['title'],
                'year': details['year'],
                'similarity': score
            })
        return similar_movies

    def generate_recommendation_explanation(self, item_cf_model, user_id, recommended_movie_id):
        """Generate a natural-language explanation for why a movie is recommended.
        
        Based on the user's history and item similarities in Item-Based CF.
        """
        # Get user history
        user_ratings = self.train_df[self.train_df['user_id'] == user_id]
        if user_ratings.empty:
            return "This movie is recommended based on overall platform popularity."
            
        # Check if recommended movie is in similarity matrix
        if recommended_movie_id not in item_cf_model.movie_id_to_idx:
            return "This movie is recommended based on its high average rating."
            
        rec_m_idx = item_cf_model.movie_id_to_idx[recommended_movie_id]
        similarities = item_cf_model.similarity_matrix[rec_m_idx]
        
        # Find user's highly rated movies (rating >= 3.5)
        highly_rated = user_ratings[user_ratings['rating'] >= 3.5]
        if highly_rated.empty:
            # Fallback to any rated movie
            highly_rated = user_ratings
            
        best_match_id = None
        max_sim = -1.0
        
        for row in highly_rated.itertuples():
            if row.movie_id in item_cf_model.movie_id_to_idx:
                user_m_idx = item_cf_model.movie_id_to_idx[row.movie_id]
                sim = similarities[user_m_idx]
                if sim > max_sim:
                    max_sim = sim
                    best_match_id = row.movie_id
                    
        if best_match_id is not None and max_sim > 0.05:
            match_details = self.get_movie_details(best_match_id)
            match_rating = float(highly_rated[highly_rated['movie_id'] == best_match_id]['rating'].iloc[0])
            
            explanation = (
                f"Recommended because you rated '{match_details['title']}' ({match_details['year']}) "
                f"a {match_rating:.1f}/5.0 stars, and these two movies share a rating similarity score "
                f"of {max_sim:.2f}."
            )
            return explanation
            
        # Fallback explanation if no significant similarity is found
        return "Recommended because it matches your general rating profile and has high interest on the platform."

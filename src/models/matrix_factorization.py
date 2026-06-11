import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.models.base_model import BaseRecommender

class RatingsDataset(Dataset):
    """Custom PyTorch Dataset for loading rating interactions."""
    def __init__(self, users, movies, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class MatrixFactorizationModel(nn.Module):
    """PyTorch Matrix Factorization Model with Biases."""
    def __init__(self, num_users, num_movies, embedding_dim=50, global_mean=3.5):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)
        
        # Initialize biases to 0
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)
        
        # Initialize embeddings with small random values (standard dev 0.05)
        nn.init.normal_(self.user_embeddings.weight, std=0.05)
        nn.init.normal_(self.movie_embeddings.weight, std=0.05)
        
        self.global_mean = nn.Parameter(torch.tensor(global_mean, dtype=torch.float32), requires_grad=False)

    def forward(self, user_indices, movie_indices):
        user_embeds = self.user_embeddings(user_indices)
        movie_embeds = self.movie_embeddings(movie_indices)
        
        # Dot product
        interaction = torch.sum(user_embeds * movie_embeds, dim=1)
        
        user_bias = self.user_biases(user_indices).squeeze(1)
        movie_bias = self.movie_biases(movie_indices).squeeze(1)
        
        # Predict: dot_product + user_bias + movie_bias + global_mean
        preds = interaction + user_bias + movie_bias + self.global_mean
        return preds

class MatrixFactorizationRecommender(BaseRecommender):
    """Wrapper class for PyTorch Matrix Factorization."""
    
    def __init__(self, embedding_dim=50, lr=0.005, weight_decay=0.01, epochs=15, batch_size=256, device=None):
        """
        Args:
            embedding_dim (int): Dimensionality of latent embeddings.
            lr (float): Learning rate.
            weight_decay (float): L2 regularization weight.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            device (str): Device to train on ('cpu' or 'cuda' or 'mps').
        """
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Matrix Factorization using device: {self.device}")
        
        # Internal states
        self.global_mean = 3.5
        self.user_means = {}
        self.movie_means = {}
        self.movie_id_to_idx = {}
        self.movie_idx_to_id = {}
        self.user_id_to_idx = {}
        self.user_idx_to_id = {}
        self.model = None
        self.rated_movies_by_user = {}  # Keep track of items to exclude in recommend()
        
    def fit(self, train_df):
        print("Fitting PyTorch Matrix Factorization model...")
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
        
        # Track already rated movies per user
        self.rated_movies_by_user = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()
        
        # Prepare datasets
        train_user_indices = train_df['user_id'].map(self.user_id_to_idx).values
        train_movie_indices = train_df['movie_id'].map(self.movie_id_to_idx).values
        train_ratings = train_df['rating'].values
        
        dataset = RatingsDataset(train_user_indices, train_movie_indices, train_ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = MatrixFactorizationModel(
            num_users=num_users, 
            num_movies=num_movies, 
            embedding_dim=self.embedding_dim, 
            global_mean=self.global_mean
        ).to(self.device)
        
        # Optimizer and loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Training loop
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for users, movies, ratings in dataloader:
                users = users.to(self.device)
                movies = movies.to(self.device)
                ratings = ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(users, movies)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(ratings)
                
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch}/{self.epochs} - Loss (MSE): {epoch_loss:.4f} - RMSE: {np.sqrt(epoch_loss):.4f}")
            
        print("Model training complete!")
        self.model.eval()
        
    def predict(self, user_id, movie_id):
        # Fallbacks for unseen elements
        if user_id not in self.user_id_to_idx:
            return self.movie_means.get(movie_id, self.global_mean)
        if movie_id not in self.movie_id_to_idx:
            return self.user_means.get(user_id, self.global_mean)
            
        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        
        # Run through model
        u_tensor = torch.tensor([u_idx], dtype=torch.long).to(self.device)
        m_tensor = torch.tensor([m_idx], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            pred = self.model(u_tensor, m_tensor).item()
            
        # Clip to valid scale
        return float(np.clip(pred, 1.0, 5.0))
        
    def recommend(self, user_id, n_items=10, exclude_rated_items=True):
        if user_id not in self.user_id_to_idx:
            # Cold start user: recommend top popular/highly rated items
            sorted_movies = sorted(self.movie_means.items(), key=lambda x: x[1], reverse=True)
            return sorted_movies[:n_items]
            
        u_idx = self.user_id_to_idx[user_id]
        rated_items = self.rated_movies_by_user.get(user_id, set())
        
        # Vectorized recommendation generation
        num_movies = len(self.movie_idx_to_id)
        u_tensor = torch.full((num_movies,), u_idx, dtype=torch.long).to(self.device)
        m_tensor = torch.arange(num_movies, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            preds = self.model(u_tensor, m_tensor).cpu().numpy()
            
        # Clip predictions to valid scale
        preds = np.clip(preds, 1.0, 5.0)
        
        if exclude_rated_items:
            # Mask out items the user has already rated in train set
            for m_id in rated_items:
                if m_id in self.movie_id_to_idx:
                    m_idx = self.movie_id_to_idx[m_id]
                    preds[m_idx] = -1.0
                    
        # Sort predictions
        top_indices = np.argsort(preds)[::-1][:n_items]
        
        recommendations = []
        for m_idx in top_indices:
            # Skip if it was excluded
            if preds[m_idx] < 0:
                continue
            movie_id = self.movie_idx_to_id[m_idx]
            recommendations.append((movie_id, float(preds[m_idx])))
            
        return recommendations

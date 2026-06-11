import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.models.base_model import BaseRecommender
from src.models.matrix_factorization import RatingsDataset

class NCFModel(nn.Module):
    """Neural Collaborative Filtering (NCF) Model combining GMF and MLP."""
    def __init__(self, num_users, num_movies, latent_dim_gmf=32, latent_dim_mlp=32, 
                 layers=[64, 32, 16, 8], global_mean=3.5):
        super(NCFModel, self).__init__()
        
        self.global_mean = nn.Parameter(torch.tensor(global_mean, dtype=torch.float32), requires_grad=False)
        
        # GMF Embeddings
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.movie_embed_gmf = nn.Embedding(num_movies, latent_dim_gmf)
        
        # MLP Embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.movie_embed_mlp = nn.Embedding(num_movies, latent_dim_mlp)
        
        # MLP Hidden Layers
        mlp_layers = []
        input_dim = latent_dim_mlp * 2  # concatenated user & movie embeddings
        for i in range(len(layers)):
            mlp_layers.append(nn.Linear(input_dim if i == 0 else layers[i-1], layers[i]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=0.2))
            
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        # Combines GMF path (latent_dim_gmf) and MLP path (last layer of MLP)
        final_input_dim = latent_dim_gmf + layers[-1]
        self.prediction_layer = nn.Linear(final_input_dim, 1)
        
        # Initialize weights
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.movie_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.movie_embed_mlp.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                
        nn.init.xavier_uniform_(self.prediction_layer.weight)

    def forward(self, user_indices, movie_indices):
        # GMF Part
        user_gmf = self.user_embed_gmf(user_indices)
        movie_gmf = self.movie_embed_gmf(movie_indices)
        phi_gmf = user_gmf * movie_gmf  # element-wise product
        
        # MLP Part
        user_mlp = self.user_embed_mlp(user_indices)
        movie_mlp = self.movie_embed_mlp(movie_indices)
        phi_mlp_input = torch.cat([user_mlp, movie_mlp], dim=-1)
        phi_mlp = self.mlp(phi_mlp_input)
        
        # Fuse GMF and MLP
        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        
        # Output prediction (offset by global mean for training stability)
        output = self.prediction_layer(fusion).squeeze()
        return output + self.global_mean

class NeuralCollabFiltering(BaseRecommender):
    """Wrapper class for PyTorch Neural Collaborative Filtering."""
    
    def __init__(self, latent_dim_gmf=32, latent_dim_mlp=32, layers=[64, 32, 16, 8],
                 lr=0.001, weight_decay=1e-5, epochs=10, batch_size=256, device=None):
        """
        Args:
            latent_dim_gmf (int): Dimensionality of GMF embeddings.
            latent_dim_mlp (int): Dimensionality of MLP embeddings.
            layers (list): Dimensions of MLP hidden layers. First layer must equal latent_dim_mlp * 2.
            lr (float): Learning rate.
            weight_decay (float): L2 regularization weight (weight decay).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            device (str): Device to train on ('cpu' or 'cuda' or 'mps').
        """
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp
        self.layers = layers
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
            
        print(f"NCF Recommender using device: {self.device}")
        
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
        print("Fitting PyTorch Neural Collaborative Filtering model...")
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
        self.model = NCFModel(
            num_users=num_users, 
            num_movies=num_movies, 
            latent_dim_gmf=self.latent_dim_gmf, 
            latent_dim_mlp=self.latent_dim_mlp,
            layers=self.layers,
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
        
        # Predict in chunks to prevent memory limits
        preds = []
        chunk_size = 1000
        with torch.no_grad():
            for i in range(0, num_movies, chunk_size):
                u_chunk = u_tensor[i:i+chunk_size]
                m_chunk = m_tensor[i:i+chunk_size]
                pred_chunk = self.model(u_chunk, m_chunk).cpu().numpy()
                preds.append(pred_chunk)
                
        preds = np.concatenate(preds)
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

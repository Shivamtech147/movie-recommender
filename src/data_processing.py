import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from tqdm import tqdm

# Constants
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DATASET_URL = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"
TAR_PATH = os.path.join(RAW_DIR, "nf_prize_dataset.tar.gz")

def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

import subprocess
import time

def download_dataset():
    """Download the Netflix Prize Dataset from Archive.org if not present."""
    ensure_directories()
    
    # Check if already fully downloaded
    target_size = 697552028
    if os.path.exists(TAR_PATH):
        size = os.path.getsize(TAR_PATH)
        if size >= target_size - 1000: # allow minor difference
            print(f"Dataset archive already exists and is complete ({size} bytes) at {TAR_PATH}. Skipping download.")
            return
        else:
            print(f"Dataset archive exists but is incomplete ({size}/{target_size} bytes). Resuming download...")
    
    print(f"Downloading Netflix Prize Dataset from {DATASET_URL} using curl (resuming if interrupted)...")
    
    max_retries = 15
    for attempt in range(1, max_retries + 1):
        try:
            # -L follows redirects
            # -C - resumes download from offset
            cmd = ["curl", "-L", "-C", "-", "-o", TAR_PATH, DATASET_URL]
            print(f"Running: {' '.join(cmd)} (Attempt {attempt}/{max_retries})")
            
            subprocess.run(cmd, check=True)
            
            # Verify file size
            if os.path.exists(TAR_PATH):
                size = os.path.getsize(TAR_PATH)
                if size >= target_size - 1000:
                    print("Download completed successfully!")
                    return
            print(f"Download finished but file size is incorrect ({os.path.getsize(TAR_PATH)} bytes). Retrying...")
        except subprocess.CalledProcessError as e:
            print(f"Curl command failed: {e}")
            if attempt < max_retries:
                print("Connection lost. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                if not os.path.exists(os.path.join(RAW_DIR, "combined_data_1.txt")):
                    raise RuntimeError("Failed to download dataset after multiple attempts.") from e

def extract_dataset():
    """Extract the downloaded tar.gz file."""
    print("Checking for extracted raw files...")
    expected_files = [
        "training_set.tar",
        "movie_titles.txt"
    ]
    all_extracted = all(os.path.exists(os.path.join(RAW_DIR, f)) for f in expected_files)
    
    if all_extracted:
        print("Raw files already extracted in raw directory.")
        return

    if not os.path.exists(TAR_PATH):
        raise FileNotFoundError(f"Dataset archive not found at {TAR_PATH} and raw files are missing.")

    print(f"Extracting {TAR_PATH} to {RAW_DIR}...")
    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting files"):
                base_name = os.path.basename(member.name)
                if not base_name:
                    continue
                member.name = base_name
                tar.extract(member, path=RAW_DIR)
        print("Extraction completed successfully!")
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise e

def parse_movie_titles():
    """Parse movie_titles.txt file and save to processed directory."""
    titles_raw_path = os.path.join(RAW_DIR, "movie_titles.txt")
    titles_processed_path = os.path.join(PROCESSED_DIR, "movie_titles.csv")
    
    if os.path.exists(titles_processed_path):
        print(f"Processed movie titles already exist at {titles_processed_path}.")
        return pd.read_csv(titles_processed_path)

    print(f"Parsing movie titles from {titles_raw_path}...")
    movies = []
    
    with open(titles_raw_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 2)
            if len(parts) >= 3:
                movie_id = int(parts[0])
                year = parts[1]
                title = parts[2]
                try:
                    year = int(year) if year != 'NULL' else np.nan
                except ValueError:
                    year = np.nan
                movies.append({
                    'movie_id': movie_id,
                    'year': year,
                    'title': title
                })
            elif len(parts) == 2:
                movie_id = int(parts[0])
                title = parts[1]
                movies.append({
                    'movie_id': movie_id,
                    'year': np.nan,
                    'title': title
                })

    df_movies = pd.DataFrame(movies)
    df_movies.to_csv(titles_processed_path, index=False)
    print(f"Saved {len(df_movies)} movie titles to {titles_processed_path}")
    return df_movies

def parse_and_sample_ratings(min_user_ratings=50, min_movie_ratings=100, 
                             num_users=10000, num_movies=2000, max_movies_to_read=4000, random_state=42):
    """Parse ratings directly from training_set.tar in memory and sample a subset."""
    processed_ratings_path = os.path.join(PROCESSED_DIR, "ratings_sample.csv")
    
    if os.path.exists(processed_ratings_path):
        print(f"Sampled ratings already exist at {processed_ratings_path}.")
        return pd.read_csv(processed_ratings_path)

    tar_path = os.path.join(RAW_DIR, "training_set.tar")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"training_set.tar not found at {tar_path}")

    print("Parsing ratings directly from training_set.tar in memory...")
    
    user_ids = []
    ratings = []
    dates = []
    movie_ids = []
    
    movies_read = 0
    
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        # Filter text files in the archive
        file_members = [m for m in members if m.isfile() and m.name.endswith('.txt')]
        # Sort members by filename to ensure consistency
        file_members.sort(key=lambda m: m.name)
        
        # Read the first max_movies_to_read movies to keep it memory-efficient and fast
        for member in tqdm(file_members[:max_movies_to_read], desc="Parsing movie files"):
            f = tar.extractfile(member)
            if f is None:
                continue
            
            content = f.read().decode('utf-8', errors='ignore')
            lines = content.strip().split('\n')
            if not lines:
                continue
                
            # First line is "MovieID:"
            movie_id_str = lines[0].strip()
            if movie_id_str.endswith(':'):
                movie_id = int(movie_id_str[:-1])
            else:
                continue
                
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    user_ids.append(int(parts[0]))
                    ratings.append(float(parts[1]))
                    dates.append(parts[2])
                    movie_ids.append(movie_id)
            
            movies_read += 1

    print(f"Read {movies_read} movies, total {len(ratings):,} rating interactions.")
    print("Converting parsed data to DataFrame...")
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'date': dates
    })
    
    # Optimize types to save memory
    df['rating'] = df['rating'].astype(np.float32)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded DataFrame. Shape: {df.shape}")
    print(f"Unique Users: {df['user_id'].nunique()}, Unique Movies: {df['movie_id'].nunique()}")

    # Apply filters
    print(f"Filtering users with >= {min_user_ratings} ratings and movies with >= {min_movie_ratings} ratings...")
    movie_counts = df['movie_id'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df_filtered = df[df['movie_id'].isin(popular_movies)]
    
    user_counts = df_filtered['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    df_filtered = df_filtered[df_filtered['user_id'].isin(active_users)]
    
    print(f"Filtered dataset shape: {df_filtered.shape}")
    print(f"Filtered Users: {df_filtered['user_id'].nunique()}, Filtered Movies: {df_filtered['movie_id'].nunique()}")

    # Sample N users and M movies to form the dense subset
    # Select top popular movies and top active users to ensure every item has adequate rating count
    print(f"Selecting top {num_movies} popular movies and top {num_users} active users...")
    
    # Get top popular movies
    movie_popularity = df_filtered['movie_id'].value_counts()
    sampled_movies = movie_popularity.head(num_movies).index.values
    
    # Filter ratings to only popular movies
    df_temp = df_filtered[df_filtered['movie_id'].isin(sampled_movies)]
    
    # Get top active users who rated these popular movies
    user_activity = df_temp['user_id'].value_counts()
    sampled_users = user_activity.head(num_users).index.values
    
    # Slice the final dataset
    df_sampled = df_temp[df_temp['user_id'].isin(sampled_users)]
    
    print(f"Final sampled dataset shape: {df_sampled.shape}")
    print(f"Final Users: {df_sampled['user_id'].nunique()}, Final Movies: {df_sampled['movie_id'].nunique()}")
    print(f"Sparsity: {100 * (1 - len(df_sampled) / (df_sampled['user_id'].nunique() * df_sampled['movie_id'].nunique())):.2f}%")
    
    # Save the sample
    df_sampled.to_csv(processed_ratings_path, index=False)
    print(f"Saved sampled ratings to {processed_ratings_path}")
    return df_sampled

def split_data(df, random_state=42):
    """Perform a user-stratified train-test split (80/20) and save split datasets."""
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Train/test splits already exist. Loading from {PROCESSED_DIR}...")
        return pd.read_csv(train_path), pd.read_csv(test_path)

    print("Performing user-stratified train-test split (80/20)...")
    # Shuffle dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Group by user_id and assign sequence index
    df_shuffled['user_rating_idx'] = df_shuffled.groupby('user_id').cumcount()
    df_shuffled['user_rating_count'] = df_shuffled.groupby('user_id')['user_id'].transform('count')
    
    # We want 20% of ratings in the test set
    is_test = df_shuffled['user_rating_idx'] < (df_shuffled['user_rating_count'] * 0.2).astype(int)
    
    # Split
    train_df = df_shuffled[~is_test].drop(columns=['user_rating_idx', 'user_rating_count'])
    test_df = df_shuffled[is_test].drop(columns=['user_rating_idx', 'user_rating_count'])
    
    # Ensure all test users are also present in training
    train_users = set(train_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    unmatched_users = test_users - train_users
    
    if unmatched_users:
        print(f"Warning: {len(unmatched_users)} users in test set are not in train set. Re-allocating ratings...")
        # For unmatched users, move one rating from test to train
        for user_id in unmatched_users:
            user_test_ratings = test_df[test_df['user_id'] == user_id]
            # Move first record
            idx_to_move = user_test_ratings.index[0]
            row_to_move = test_df.loc[[idx_to_move]]
            train_df = pd.concat([train_df, row_to_move])
            test_df = test_df.drop(idx_to_move)

    print(f"Train Set Shape: {train_df.shape}, Unique Users: {train_df['user_id'].nunique()}, Unique Movies: {train_df['movie_id'].nunique()}")
    print(f"Test Set Shape: {test_df.shape}, Unique Users: {test_df['user_id'].nunique()}, Unique Movies: {test_df['movie_id'].nunique()}")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Train/test splits saved successfully!")
    return train_df, test_df

def run_pipeline():
    """Run full data preparation pipeline."""
    ensure_directories()
    download_dataset()
    extract_dataset()
    parse_movie_titles()
    df_sample = parse_and_sample_ratings(
        min_user_ratings=50, 
        min_movie_ratings=100, 
        num_users=5000, 
        num_movies=1500
    )
    split_data(df_sample)
    print("Data processing pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()

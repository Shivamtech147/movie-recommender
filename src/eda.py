import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for premium aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 200
})

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

def ensure_directories():
    os.makedirs(PLOT_DIR, exist_ok=True)

def run_eda():
    ensure_directories()
    
    ratings_path = os.path.join(PROCESSED_DIR, "ratings_sample.csv")
    movies_path = os.path.join(PROCESSED_DIR, "movie_titles.csv")
    
    if not os.path.exists(ratings_path):
        print(f"Sampled ratings not found at {ratings_path}. Run data_processing first.")
        return

    print("Loading datasets for EDA...")
    df_ratings = pd.read_csv(ratings_path)
    df_movies = pd.read_csv(movies_path)
    
    # Merge titles for analysis
    df_ratings_merged = df_ratings.merge(df_movies, on='movie_id', how='left')
    
    # 1. Summary Statistics
    num_ratings = len(df_ratings)
    num_users = df_ratings['user_id'].nunique()
    num_movies = df_ratings['movie_id'].nunique()
    
    possible_interactions = num_users * num_movies
    sparsity = (1 - (num_ratings / possible_interactions)) * 100
    density = 100 - sparsity
    
    rating_mean = df_ratings['rating'].mean()
    rating_std = df_ratings['rating'].std()
    rating_median = df_ratings['rating'].median()
    
    # Top movies by ratings count
    top_rated_movies = df_ratings_merged['title'].value_counts().head(10).reset_index()
    top_rated_movies.columns = ['title', 'rating_count']
    
    # Top movies by average rating (min 100 ratings)
    movie_stats = df_ratings_merged.groupby('title').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    )
    best_movies = movie_stats[movie_stats['rating_count'] >= 100].sort_values(by='avg_rating', ascending=False).head(10).reset_index()

    stats = {
        'num_ratings': int(num_ratings),
        'num_users': int(num_users),
        'num_movies': int(num_movies),
        'sparsity_percent': float(sparsity),
        'density_percent': float(density),
        'rating_mean': float(rating_mean),
        'rating_std': float(rating_std),
        'rating_median': float(rating_median),
        'top_rated_movies_count': top_rated_movies.to_dict(orient='records'),
        'best_movies_avg': best_movies.to_dict(orient='records')
    }
    
    # Save statistics
    stats_path = os.path.join(OUTPUT_DIR, "eda_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved summary statistics to {stats_path}")
    
    print("\n--- DATASET SUMMARY STATISTICS ---")
    print(f"Total Ratings: {num_ratings:,}")
    print(f"Unique Users:  {num_users:,}")
    print(f"Unique Movies: {num_movies:,}")
    print(f"Sparsity:      {sparsity:.4f}%")
    print(f"Density:       {density:.4f}%")
    print(f"Average Rating:{rating_mean:.3f} ± {rating_std:.3f}")
    print(f"Median Rating: {rating_median}")
    print("---------------------------------\n")

    # 2. Plots
    
    # Plot 1: Rating Distribution
    plt.figure(figsize=(8, 5))
    rating_counts = df_ratings['rating'].value_counts().sort_index()
    colors = sns.color_palette("viridis", len(rating_counts))
    bars = plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', width=0.6)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (num_ratings * 0.01), 
                 f'{height:,}\n({height/num_ratings*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    plt.title('Distribution of Movie Ratings', pad=15)
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Count')
    plt.xticks(range(1, 6))
    plt.ylim(0, max(rating_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rating_distribution.png"))
    plt.close()
    
    # Plot 2: User Activity Distribution (Ratings per User)
    plt.figure(figsize=(10, 5))
    user_ratings = df_ratings.groupby('user_id').size()
    sns.histplot(user_ratings, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(user_ratings.mean(), color='red', linestyle='--', label=f'Mean: {user_ratings.mean():.1f}')
    plt.axvline(user_ratings.median(), color='green', linestyle='-', label=f'Median: {user_ratings.median():.1f}')
    plt.title('Distribution of Ratings per User (User Activity)', pad=15)
    plt.xlabel('Number of Ratings Given')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "user_activity.png"))
    plt.close()

    # Plot 3: Movie Popularity Distribution (Ratings per Movie)
    plt.figure(figsize=(10, 5))
    movie_ratings = df_ratings.groupby('movie_id').size()
    sns.histplot(movie_ratings, bins=50, kde=True, color='salmon', edgecolor='black')
    plt.axvline(movie_ratings.mean(), color='red', linestyle='--', label=f'Mean: {movie_ratings.mean():.1f}')
    plt.axvline(movie_ratings.median(), color='green', linestyle='-', label=f'Median: {movie_ratings.median():.1f}')
    plt.title('Distribution of Ratings per Movie (Movie Popularity)', pad=15)
    plt.xlabel('Number of Ratings Received')
    plt.ylabel('Number of Movies')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "movie_popularity.png"))
    plt.close()

    # Plot 4: Cumulative Ratings Curve (Lorenz-like curve for popularity)
    plt.figure(figsize=(8, 6))
    sorted_movie_counts = movie_ratings.sort_values(ascending=False).values
    cumulative_ratings = np.cumsum(sorted_movie_counts)
    cumulative_percent = (cumulative_ratings / cumulative_ratings[-1]) * 100
    movie_percent = (np.arange(1, len(movie_ratings) + 1) / len(movie_ratings)) * 100
    
    plt.plot(movie_percent, cumulative_percent, label='Dataset Cumulative Ratings', color='purple', linewidth=2.5)
    plt.plot([0, 100], [0, 100], linestyle='--', color='grey', label='Equal Popularity Baseline')
    
    # Annotate Pareto principle (e.g., 20% of movies account for X% of ratings)
    pareto_idx = np.abs(movie_percent - 20).argmin()
    pareto_val = cumulative_percent[pareto_idx]
    plt.plot([20, 20], [0, pareto_val], color='red', linestyle=':')
    plt.plot([0, 20], [pareto_val, pareto_val], color='red', linestyle=':')
    plt.scatter(20, pareto_val, color='red', zorder=5)
    plt.annotate(f'Top 20% Movies\naccount for {pareto_val:.1f}% ratings', 
                 xy=(20, pareto_val), xytext=(30, pareto_val - 15),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
                 
    plt.title('Concentration of Ratings (Cumulative Popularity)', pad=15)
    plt.xlabel('Percentage of Movies (Sorted by Popularity)')
    plt.ylabel('Percentage of Total Ratings')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cumulative_ratings.png"))
    plt.close()
    
    print("EDA Visualizations generated and saved to outputs/plots/!")

if __name__ == "__main__":
    run_eda()

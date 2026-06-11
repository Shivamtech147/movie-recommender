import os
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Netflix Personalization Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-mode aesthetic
st.markdown("""
<style>
    /* Main body background */
    .stApp {
        background-color: #0F0F13;
        color: #E2E2E9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #16161D;
        border-right: 1px solid #23232F;
    }
    
    /* Header/Title banner */
    .main-title {
        background: linear-gradient(135deg, #FF1E27 0%, #B00612 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1A1A24;
        border: 1px solid #282836;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #FF1E27;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #FF1E27;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #8D8D9D;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Custom container lists */
    .recs-container {
        background-color: #161622;
        border-left: 4px solid #FF1E27;
        border-radius: 4px 8px 8px 4px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .explanation-text {
        font-style: italic;
        color: #9A9AA4;
        font-size: 13.5px;
        margin-top: 5px;
    }
    
    /* Title colors */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Load data helper
@st.cache_data
def load_metadata():
    movie_titles = pd.read_csv(os.path.join(DATA_DIR, "movie_titles.csv"))
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    # Load EDA statistics
    with open(os.path.join(OUTPUT_DIR, "eda_stats.json"), "r") as f:
        eda_stats = json.load(f)
        
    return movie_titles, train_df, eda_stats

# Load models helper
@st.cache_resource
def load_model(model_name):
    if model_name == "Item-Based CF":
        path = os.path.join(MODELS_DIR, "item_cf.pkl")
    elif model_name == "Matrix Factorization":
        path = os.path.join(MODELS_DIR, "matrix_factorization.pkl")
    else:  # Neural CF
        path = os.path.join(MODELS_DIR, "ncf.pkl")
        
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def main():
    # Verify execution outputs exist
    if not os.path.exists(os.path.join(OUTPUT_DIR, "eda_stats.json")):
        st.error("Pipeline has not been executed yet. Run `python3 run_pipeline.py` in the terminal first to process data and train models.")
        return
        
    movie_titles, train_df, eda_stats = load_metadata()
    
    # Import recommendation manager
    from src.recommendation import RecommendationManager
    rec_manager = RecommendationManager(train_df, movie_titles)
    
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center;'>🎬 Netflix Engine</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    nav_option = st.sidebar.radio(
        "Navigation",
        ["📊 Exploratory Data Analysis", "🎯 Personalized Discovery", "🔍 Content Explorer", "📈 Model Benchmarks"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    model_option = st.sidebar.selectbox(
        "Active Model",
        ["Item-Based CF", "Matrix Factorization", "Neural CF (NCF)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Team Participants**: Max 02")
    st.sidebar.markdown("Netflix Prize Personalization Project")
    
    # Main Banner
    st.markdown("""
    <div class="main-title">
        <h1 style="margin: 0; font-size: 32px; font-weight: 800;">Personalized Content Discovery</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 16px;">Next-Generation Recommendation System Powered by Latent Factor Modeling & Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Actions
    if nav_option == "📊 Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        st.write("Understand the shape, distribution, and sparsity patterns of our movie rating interaction dataset.")
        
        # Summary statistics layout
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_stats["num_ratings"]:,}</div><div class="metric-label">Ratings</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_stats["num_users"]:,}</div><div class="metric-label">Users</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_stats["num_movies"]:,}</div><div class="metric-label">Movies</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_stats["sparsity_percent"]:.2f}%</div><div class="metric-label">Sparsity</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{eda_stats["rating_mean"]:.2f}★</div><div class="metric-label">Avg Rating</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plots Grid
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.markdown("### Movie Rating Distribution")
            rating_dist_img = os.path.join(OUTPUT_DIR, "plots", "rating_distribution.png")
            if os.path.exists(rating_dist_img):
                st.image(rating_dist_img, use_container_width=True)
            st.write("**Insight**: The ratings distribution shows that users lean positive, with 4-star and 3-star ratings being the most common. Ratings of 1 and 2 stars are relatively rare, which matches typical user feedback bias on streaming platforms.")
            
        with col_plot2:
            st.markdown("### Concentration of Ratings (Lorenz Curve)")
            cum_ratings_img = os.path.join(OUTPUT_DIR, "plots", "cumulative_ratings.png")
            if os.path.exists(cum_ratings_img):
                st.image(cum_ratings_img, use_container_width=True)
            st.write("**Insight**: Shows a significant popularity concentration (Pareto principle). The top 20% of movies account for a dominant share of all ratings. This highlights the business need for active discovery models that can surface long-tail content to users.")
            
        col_plot3, col_plot4 = st.columns(2)
        with col_plot3:
            st.markdown("### User Activity Distribution")
            user_act_img = os.path.join(OUTPUT_DIR, "plots", "user_activity.png")
            if os.path.exists(user_act_img):
                st.image(user_act_img, use_container_width=True)
            st.write("**Insight**: The distribution shows user activity levels. Active users provide multiple interaction anchors, which provides rich preference signals for collaborative filtering models.")
            
        with col_plot4:
            st.markdown("### Movie Popularity Distribution")
            movie_pop_img = os.path.join(OUTPUT_DIR, "plots", "movie_popularity.png")
            if os.path.exists(movie_pop_img):
                st.image(movie_pop_img, use_container_width=True)
            st.write("**Insight**: Popular movies receive hundreds of ratings, providing highly dense item vectors. Surfacing tail items with few ratings remains the core challenge for SVD and content hybrid methods.")
            
    elif nav_option == "🎯 Personalized Discovery":
        st.header("Personalized Recommendation Hub")
        st.write("Enter a user's ID to fetch their watch history, generate personalized content recommendations, and inspect explains.")
        
        # Load the selected model
        model = load_model(model_option)
        if model is None:
            st.warning(f"Failed to load trained weights for {model_option}. Ensure you have run the training pipeline.")
            return
            
        # Example user IDs from training
        example_users = list(train_df['user_id'].unique()[:10])
        
        col_user, col_go = st.columns([4, 1])
        with col_user:
            selected_user = st.selectbox("Select a Sample User:", example_users)
            user_input = st.text_input("Or enter a custom User ID:", str(selected_user))
            
        try:
            user_id = int(user_input)
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
            return
            
        # Run recommendations
        with st.spinner(f"Computing Top-10 recommendations using {model_option}..."):
            recs = rec_manager.generate_personalized_recommendations(model, user_id, n_items=10)
            history = rec_manager.get_user_history(user_id, n_items=8)
            
        col_hist, col_recs = st.columns([1, 1])
        
        with col_hist:
            st.markdown("### 📜 User Rating History")
            st.write("Top movies this user rated in the training set:")
            if history:
                df_hist = pd.DataFrame(history)
                # Display beautifully
                for row in df_hist.itertuples():
                    stars = "★" * int(np.round(row.rating)) + "☆" * (5 - int(np.round(row.rating)))
                    st.markdown(f"""
                    <div style='background-color:#1E1E28; padding:10px; border-radius:6px; margin-bottom:8px;'>
                        <strong style='color:white;'>{row.title}</strong> ({int(row.year) if not np.isnan(row.year) else 'N/A'})<br>
                        <span style='color:#FFC107;'>{stars}</span> ({row.rating:.1f}/5.0) | <small style='color:#8D8D9D;'>{row.date}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No historical records found (New User).")
                
        with col_recs:
            st.markdown(f"### 🎯 Top-10 Recommendations ({model_option})")
            st.write("Personalized items predicted to match user interest:")
            
            # Load Item-Based CF model specifically for explanations (since it contains similarity matrices)
            cf_model = load_model("Item-Based CF")
            
            for i, rec in enumerate(recs):
                stars_val = rec['predicted_rating']
                stars = "★" * int(np.round(stars_val)) + "☆" * (5 - int(np.round(stars_val)))
                
                # Get explanation
                explanation = "Popular content recommendation."
                if rec['score_type'] == 'personalized' and cf_model is not None:
                    explanation = rec_manager.generate_recommendation_explanation(cf_model, user_id, rec['movie_id'])
                    
                st.markdown(f"""
                <div class="recs-container">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:16px; font-weight:bold; color:white;">{i+1}. {rec['title']} ({int(rec['year']) if not np.isnan(rec['year']) else 'N/A'})</span>
                        <span style="color:#FF1E27; font-weight:bold; font-size:15px;">{rec['predicted_rating']:.2f} ★</span>
                    </div>
                    <div class="explanation-text">💡 {explanation}</div>
                </div>
                """, unsafe_allow_html=True)
                
    elif nav_option == "🔍 Content Explorer":
        st.header("Movie Similarity Explorer")
        st.write("Understand the content space. Search for a movie and discover similar content in the rating similarity space (Item-Based CF) and the latent embedding space (Matrix Factorization/NCF).")
        
        # Load models
        cf_model = load_model("Item-Based CF")
        mf_model = load_model("Matrix Factorization")
        ncf_model = load_model("Neural CF (NCF)")
        
        # Select target movie
        available_movies = sorted(rec_manager.movie_titles_df['title'].unique())
        selected_movie_title = st.selectbox("Search and Select a Movie:", available_movies, index=available_movies.index("Toy Story") if "Toy Story" in available_movies else 0)
        
        target_movie_row = rec_manager.movie_titles_df[rec_manager.movie_titles_df['title'] == selected_movie_title]
        if target_movie_row.empty:
            st.error("Movie not found.")
            return
            
        movie_id = int(target_movie_row.iloc[0]['movie_id'])
        movie_year = target_movie_row.iloc[0]['year']
        st.subheader(f"Analyzing Similar Content to: '{selected_movie_title}' ({int(movie_year) if not np.isnan(movie_year) else 'N/A'})")
        
        col_cf_sim, col_latent_sim = st.columns(2)
        
        with col_cf_sim:
            st.markdown("### 📊 Rating-Based Similarities (Item-CF)")
            st.write("Movies co-rated in similar patterns by users (Adjusted Cosine):")
            if cf_model is not None:
                sim_cf = rec_manager.get_similar_movies_item_cf(cf_model, movie_id, n_items=8)
                if sim_cf:
                    for i, item in enumerate(sim_cf):
                        st.markdown(f"""
                        <div style='background-color:#161622; padding:12px; border-radius:6px; margin-bottom:8px; border-left:3px solid #FFC107;'>
                            <strong>{i+1}. {item['title']}</strong> ({int(item['year']) if not np.isnan(item['year']) else 'N/A'})<br>
                            <span style='color:#FFC107;'>Similarity: {item['similarity']:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No similar movies found in ratings space.")
            else:
                st.write("Item-Based CF model not loaded.")
                
        with col_latent_sim:
            st.markdown("### 🔮 Latent Semantic Similarities (Matrix Factorization)")
            st.write("Movies grouped closely in the 50-dimensional latent SVD space:")
            if mf_model is not None:
                sim_mf = rec_manager.get_similar_movies_latent(mf_model, movie_id, n_items=8)
                if sim_mf:
                    for i, item in enumerate(sim_mf):
                        st.markdown(f"""
                        <div style='background-color:#161622; padding:12px; border-radius:6px; margin-bottom:8px; border-left:3px solid #00D2FF;'>
                            <strong>{i+1}. {item['title']}</strong> ({int(item['year']) if not np.isnan(item['year']) else 'N/A'})<br>
                            <span style='color:#00D2FF;'>Embedding Cosine Sim: {item['similarity']:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No similar movies found in latent space.")
            else:
                st.write("Matrix Factorization model not loaded.")
                
    elif nav_option == "📈 Model Benchmarks":
        st.header("Recommendation Model Benchmark Arena")
        st.write("Compare the prediction quality (RMSE, MAE) and ranking quality (MAP@10) across models.")
        
        # Load comparison metrics
        comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
        if os.path.exists(comp_path):
            df_comp = pd.read_csv(comp_path, index_index=False if 'Unnamed: 0' not in pd.read_csv(comp_path).columns else True)
            if 'Unnamed: 0' in df_comp.columns:
                df_comp = df_comp.rename(columns={'Unnamed: 0': 'Model'})
                
            st.dataframe(df_comp.style.format({
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'MAP@10': '{:.4f}',
                'NDCG@10': '{:.4f}',
                'Precision@10': '{:.4f}',
                'Recall@10': '{:.4f}',
                'Train Time (s)': '{:.2f}',
                'Eval Time (s)': '{:.2f}'
            }), use_container_width=True)
            
            # Plot comparisons
            st.markdown("### Performance Comparison Visualizations")
            col_plt1, col_plt2 = st.columns(2)
            
            with col_plt1:
                st.markdown("#### Rating Prediction Error (RMSE) - Lower is Better")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_comp, x='Model', y='RMSE', hue='Model', palette='rocket', legend=False, ax=ax)
                ax.set_ylim(0.7, 1.1)
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
            with col_plt2:
                st.markdown("#### Recommendation Ranking Accuracy (MAP@10) - Higher is Better")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df_comp, x='Model', y='MAP@10', hue='Model', palette='mako', legend=False, ax=ax)
                ax.set_ylim(0.0, max(df_comp['MAP@10']) * 1.2)
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
            st.markdown("""
            ### 📝 Key Engineering Observations & Trade-offs
            1. **RMSE vs MAP@10 Trade-off**:
               Predicting rating magnitudes (RMSE) and ranking recommendations (MAP@10) are different objectives. A model can achieve a lower RMSE (e.g. Matrix Factorization predicting rating levels accurately) but might have a similar or slightly lower MAP@10 compared to collaborative filtering baselines due to bias patterns. Surfacing high-utility items requires models designed for ranking.
            2. **Training & Inference Complexity**:
               * **Item-CF** requires zero training but calculating predictions during inference requires dot products across item matrices. It scales quadratically with items ($M^2$), making it difficult for large catalogs.
               * **Matrix Factorization (SVD)** compresses users and items into a fixed embedding size (e.g., 50). Inference is extremely fast (dot product of user and movie vectors), making it highly scalable for real-time deployment.
               * **Neural CF (NCF)** has high model capacity due to MLP layers, allowing it to capture non-linear relationships. However, it takes longer to train and requires forward passes through dense layers for inference.
            """)
        else:
            st.warning("Comparison metrics CSV not found. Make sure the run_pipeline.py script has completed successfully.")

if __name__ == "__main__":
    main()

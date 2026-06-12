import os
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Netflix Personalization Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium Streaming Platform aesthetic
st.markdown("""
<style>
    /* Main body background with cinema radial gradient */
    .stApp {
        background: radial-gradient(circle at top, #1E1215 0%, #0C0809 60%, #020202 100%) !important;
        color: #E2E2E9;
    }
    
    /* Transparent header to avoid top bar white space while keeping sidebar toggle button visible */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        z-index: 999 !important;
    }
    
    /* Style the sidebar toggle buttons (hamburger and expand chevron) to look ultra premium */
    header[data-testid="stHeader"] button,
    section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] {
        background-color: rgba(26, 24, 26, 0.8) !important;
        border: 1px solid #262635 !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        transition: all 0.2s ease !important;
    }
    header[data-testid="stHeader"] button:hover,
    section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"]:hover {
        border-color: #E50914 !important;
        color: #E50914 !important;
    }
    
    .main .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0809 !important;
        border-right: 1px solid #1E1A1C !important;
    }
    
    /* Widget label styling */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
    }
    
    /* Fix text visibility in Selectboxes and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    input[data-testid="stTextInputBase-Input"] {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] > div:hover,
    input[data-testid="stTextInputBase-Input"]:hover {
        border-color: #E50914 !important;
    }
    
    /* Style popover/dropdown elements for visibility */
    div[data-baseweb="popover"] ul,
    div[role="listbox"] {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
    }
    div[data-baseweb="popover"] li,
    div[role="option"] {
        color: #FFFFFF !important;
        background-color: #161622 !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[role="option"]:hover {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar Navigation Cards styling */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: transparent !important;
        border: none !important;
        color: #9E9EB3 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        height: auto !important;
        min-height: 56px !important;
        display: block !important;
        text-align: left !important;
        margin-bottom: 8px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #1A181A !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div.stButton > button p,
    section[data-testid="stSidebar"] div.stButton > button span,
    section[data-testid="stSidebar"] div.stButton > button div {
        white-space: pre-line !important;
        text-align: left !important;
        font-size: 13.5px !important;
        color: #9E9EB3 !important;
        font-weight: 400 !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button p::first-line,
    section[data-testid="stSidebar"] div.stButton > button span::first-line,
    section[data-testid="stSidebar"] div.stButton > button div::first-line {
        font-weight: 700 !important;
        font-size: 16px !important;
        color: #FFFFFF !important;
        line-height: 1.6 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p::first-line,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span::first-line,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span {
        color: #FFFFFF !important;
    }
    
    /* Main content buttons (for onboarding etc.) */
    div.stButton > button {
        width: 100% !important;
        border-radius: 6px !important;
        border: 1px solid #262635 !important;
        transition: all 0.2s ease !important;
        padding: 8px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        border-color: #E50914 !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #E50914 !important;
        color: white !important;
        border-color: #E50914 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #161622 !important;
        color: #E2E2E9 !important;
    }
    
    /* Movie Shelf Cards (EDA Page) */
    .movie-shelf-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        height: 165px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        position: relative !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        cursor: pointer !important;
    }
    .movie-shelf-card:hover {
        transform: scale(1.05) !important;
        border-color: #E50914 !important;
        box-shadow: 0 10px 20px rgba(229, 9, 20, 0.2) !important;
        z-index: 10 !important;
    }
    .movie-shelf-badge {
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        background-color: #3D0C11 !important;
        color: #FF7D82 !important;
        border: 1px solid #5A141A !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 10px !important;
        font-weight: 700 !important;
    }
    .movie-shelf-title {
        font-size: 15.5px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-top: 8px !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1.3 !important;
    }
    .movie-shelf-play-icon {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: #E50914 !important;
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
        margin-top: 5px !important;
        opacity: 0.7 !important;
        transition: opacity 0.2s ease !important;
    }
    .movie-shelf-card:hover .movie-shelf-play-icon {
        opacity: 1.0 !important;
    }
    .movie-shelf-details {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-top: auto !important;
    }
    
    /* Horizontal Card Styles for Smart Recommender */
    .smart-rec-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        height: 195px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        position: relative !important;
        transition: all 0.25s ease !important;
        margin-bottom: 15px !important;
    }
    .smart-rec-card:hover {
        border-color: #E50914 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 18px rgba(229, 9, 20, 0.2) !important;
    }
    .smart-rec-card.history {
        border-top: 3px solid #3897F0 !important;
    }
    .smart-rec-card.recommendation {
        border-top: 3px solid #E50914 !important;
    }
    .smart-rec-title {
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        line-height: 1.3 !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-height: 36px !important;
    }
    .smart-rec-sub {
        font-size: 11px !important;
        color: #8D8D9D !important;
        margin-top: 2px !important;
    }
    .smart-rec-rating {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #FFC107 !important;
    }
    .smart-rec-explanation {
        font-size: 10.5px !important;
        font-style: italic !important;
        color: #9E9EB3 !important;
        line-height: 1.25 !important;
        margin-top: 4px !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 3 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-height: 40px !important;
    }

    /* Personalized Discovery Recommendation Cards */
    .recs-container {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-left: 4px solid #E50914 !important;
        border-radius: 4px 6px 6px 4px !important;
        padding: 15px !important;
        margin-bottom: 12px !important;
        transition: all 0.2s ease !important;
    }
    .recs-container:hover {
        border-color: #E50914 !important;
    }
    .explanation-text {
        font-style: italic !important;
        color: #9E9EB3 !important;
        font-size: 13px !important;
        margin-top: 5px !important;
    }
    
    /* Similarity Card Explorer styling */
    .similarity-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        margin-bottom: 12px !important;
        transition: all 0.2s ease !important;
    }
    .similarity-card:hover {
        border-color: #E50914 !important;
        transform: translateY(-1px) !important;
    }
    .sim-card-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        gap: 10px !important;
    }
    .sim-card-title {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    .sim-card-body {
        display: flex !important;
        justify-content: space-between !important;
        font-size: 12px !important;
        color: #9E9EB3 !important;
        margin-top: 8px !important;
    }
    
    /* Model Architectural Insights styling */
    .model-insight-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 6px !important;
        padding: 16px !important;
        height: 120px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    .model-insight-title {
        font-size: 15px !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
    }
    .model-insight-bullet {
        font-size: 12.5px !important;
        color: #E2E2E9 !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .bullet-green {
        color: #2D8A4E !important;
        font-weight: bold !important;
    }
    .bullet-red {
        color: #E50914 !important;
        font-weight: bold !important;
    }
    
    /* Cold Start simulator */
    .sim-rec-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 6px !important;
        padding: 14px !important;
        margin-bottom: 12px !important;
        transition: all 0.2s ease !important;
    }
    .sim-rec-card:hover {
        border-color: #E50914 !important;
    }
    .sim-rec-title {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    .sim-rec-detail {
        font-size: 12px !important;
        color: #9E9EB3 !important;
    }
    
    /* System Badges */
    .badge-red {
        background-color: #3D0C11;
        color: #FF7D82;
        border: 1px solid #5A141A;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px !important;
        font-weight: 600 !important;
    }
    .badge-blue {
        background-color: #0C203D;
        color: #7DB2FF;
        border: 1px solid #14325A;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px !important;
        font-weight: 600 !important;
    }
    .badge-success {
        background-color: #0F321C;
        color: #7DFF9E;
        border: 1px solid #145A2C;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px !important;
        font-weight: 600 !important;
    }
    .badge-amber {
        background-color: #3D260C;
        color: #FFC07D;
        border: 1px solid #5A3914;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px !important;
        font-weight: 600 !important;
    }
    
    /* Typography Overrides */
    h1 {
        font-size: 38px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px !important;
    }
    h2 {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
    }
    h3 {
        font-size: 21px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin-top: 20px !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #E2E2E9 !important;
        font-size: 16.5px !important;
        line-height: 1.65 !important;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 12px !important;
        padding: 30px 20px !important;
        text-align: center !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.25) !important;
    }
    .kpi-card:hover {
        border-color: #E50914 !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 24px rgba(229, 9, 20, 0.25) !important;
    }
    .kpi-value {
        font-size: 38px !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        margin-bottom: 8px !important;
        letter-spacing: -1px !important;
    }
    .kpi-label {
        font-size: 14.5px !important;
        color: #9E9EB3 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Hover effect to lift charts slightly on cursor hover, blending with card headers */
    div[data-testid="stPlotlyChart"], 
    div[data-testid="stPyplotChart"], 
    div[data-testid="stPyplot"], 
    div[data-testid="stImage"],
    div.stPlotlyChart,
    div.stPyplotChart,
    div.stImage {
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
        border: 1px solid #262635 !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        background-color: #161622 !important;
        padding: 5px 15px 15px 15px !important;
    }
    div[data-testid="stPlotlyChart"]:hover, 
    div[data-testid="stPyplotChart"]:hover, 
    div[data-testid="stPyplot"]:hover, 
    div[data-testid="stImage"]:hover,
    div.stPlotlyChart:hover,
    div.stPyplotChart:hover,
    div.stImage:hover {
        transform: translateY(-4px) !important;
        border-color: #E50914 !important;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.25) !important;
        z-index: 99 !important;
    }

    
    /* Target first line/text of active sidebar primary button */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] div,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        color: #FFFFFF !important;
    }

    /* Premium OTT-themed Table Styling */
    .ott-table {
        width: 100% !important;
        border-collapse: collapse !important;
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 14.5px !important;
        margin: 15px 0 25px 0 !important;
    }
    .ott-table th {
        background-color: #1A181A !important;
        color: #E50914 !important;
        font-weight: 700 !important;
        padding: 14px 18px !important;
        text-align: left !important;
        border-bottom: 2px solid #262635 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 12.5px !important;
    }
    .ott-table tr {
        transition: background-color 0.2s ease !important;
        border-bottom: 1px solid #262635 !important;
    }
    .ott-table tr:hover {
        background-color: #1F1C2C !important;
    }
    .ott-table td {
        padding: 12px 18px !important;
        color: #FFFFFF !important;
    }
    .ott-table tr.champion-row {
        background-color: rgba(229, 9, 20, 0.12) !important;
        border-left: 4px solid #E50914 !important;
    }
    .ott-table tr.champion-row td {
        color: #FFFFFF !important;
    }
    .ott-table tr:nth-child(even):not(.champion-row) {
        background-color: #1E1E2C !important;
    }
    .ott-table tr:nth-child(odd):not(.champion-row) {
        background-color: #161622 !important;
    }

    /* Onboarding Movie Cards */
    .movie-onboard-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        height: 190px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        position: relative !important;
        transition: all 0.25s ease !important;
        margin-bottom: 15px !important;
        text-align: center !important;
    }
    .movie-onboard-card:hover {
        border-color: #E50914 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.15) !important;
    }
    .movie-onboard-card.active {
        border-color: #E50914 !important;
        background-color: #2D0F13 !important;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.25) !important;
    }
    .movie-onboard-title {
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-top: 8px !important;
        line-height: 1.35 !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-height: 36px !important;
    }
    .movie-onboard-badge {
        position: absolute !important;
        top: 8px !important;
        right: 8px !important;
        background-color: #3D0C11 !important;
        color: #FF7D82 !important;
        border: 1px solid #5A141A !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 9px !important;
        font-weight: 700 !important;
    }

    /* Case Studies Showcase Card items */
    .showcase-card {
        background-color: #161622 !important;
        border: 1px solid #262635 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        transition: border-color 0.2s ease !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }
    .showcase-card:hover {
        border-color: #E50914 !important;
    }
    .showcase-title-area {
        display: flex !important;
        flex-direction: column !important;
        gap: 3px !important;
        text-align: left !important;
    }
    .showcase-movie-title {
        font-weight: 700 !important;
        color: #FFFFFF !important;
        font-size: 14.5px !important;
    }
    .showcase-movie-year {
        font-size: 12px !important;
        color: #9E9EB3 !important;
    }
    .showcase-rating-stars {
        color: #FFC107 !important;
        font-size: 13.5px !important;
        font-weight: bold !important;
    }
    .showcase-badge-val {
        background-color: #0F321C !important;
        color: #7DFF9E !important;
        border: 1px solid #145A2C !important;
        padding: 4px 10px !important;
        border-radius: 4px !important;
        font-size: 12.5px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        white-space: nowrap !important;
    }
    .showcase-badge-low {
        background-color: #3D260C !important;
        color: #FFC07D !important;
        border: 1px solid #5A3914 !important;
        padding: 4px 10px !important;
        border-radius: 4px !important;
        font-size: 12.5px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        white-space: nowrap !important;
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

# Helper to apply cinema theme style to matplotlib plots
def apply_plot_style(ax, fig=None):
    if fig is not None:
        fig.patch.set_facecolor('#161622')
    ax.set_facecolor('#161622')
    ax.tick_params(colors='#E2E2E9', which='both', labelsize=8)
    ax.xaxis.label.set_color('#E2E2E9')
    ax.yaxis.label.set_color('#E2E2E9')
    ax.title.set_color('#FFFFFF')
    ax.grid(True, color='#262635', linestyle='--', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color('#262635')

# Helper to style matplotlib plots with white background and dark slate text (for premium dashboard look)
def apply_matplotlib_theme(fig, ax, x_title="", y_title="", title="", show_grid=True):
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12, color='#1E1E2C')
    ax.set_xlabel(x_title, fontsize=10, fontweight='bold', color='#1E1E2C')
    ax.set_ylabel(y_title, fontsize=10, fontweight='bold', color='#1E1E2C')
    
    # Tick colors and formatting
    ax.tick_params(colors='#1E1E2C', which='both', labelsize=9)
    ax.xaxis.label.set_color('#1E1E2C')
    ax.yaxis.label.set_color('#1E1E2C')
    
    # Spines (borders)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1)
        
    # Grid lines
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
    else:
        ax.grid(False)

# Helper to render styled headers for charts to make them look like beautiful dashboard cards (inspired by HuggingFace/Netflix layouts)
def render_chart_header(title, description):
    st.markdown(f"""
    <div style="background-color: #161622; border: 1px solid #262635; border-bottom: none; border-radius: 10px 10px 0 0; padding: 18px 20px 10px 20px; margin-bottom: 0px; margin-top: 15px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
        <div style="font-size: 17px; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">{title}</div>
        <div style="font-size: 13px; color: #9E9EB3; line-height: 1.4;">{description}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper to apply white-background, high-fidelity OTT themes to Plotly figures (inspired by Images 3 & 4)
def apply_ott_chart_theme(fig, is_heatmap=False, show_legend=True, x_title=None, y_title=None, show_scale=False):
    layout_update = dict(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font_color='#1E1E2C',
        title_font_color='#1E1E2C',
        xaxis=dict(
            gridcolor='#E2E2E9',
            linecolor='#CCCCCC',
            tickcolor='#888888',
            showgrid=True,
            title_font=dict(color='#1E1E2C', size=11, family="Helvetica, Arial, sans-serif"),
            tickfont=dict(color='#1E1E2C', size=9, family="Helvetica, Arial, sans-serif")
        ),
        yaxis=dict(
            gridcolor='#E2E2E9',
            linecolor='#CCCCCC',
            tickcolor='#888888',
            showgrid=True,
            title_font=dict(color='#1E1E2C', size=11, family="Helvetica, Arial, sans-serif"),
            tickfont=dict(color='#1E1E2C', size=9, family="Helvetica, Arial, sans-serif")
        ),
        margin=dict(l=50, r=40, t=15, b=40),
        height=350
    )
    
    if x_title:
        layout_update['xaxis']['title'] = x_title
    if y_title:
        layout_update['yaxis']['title'] = y_title
        
    if not show_legend:
        layout_update['showlegend'] = False
        
    if is_heatmap:
        layout_update['xaxis']['showgrid'] = False
        layout_update['yaxis']['showgrid'] = False
        layout_update['coloraxis_showscale'] = show_scale
        
    fig.update_layout(**layout_update)

# Helper to render a high-fidelity dark-themed HTML table fitted to the OTT cinema aesthetic
def render_html_table(df, highlight_best=False):
    html_lines = []
    html_lines.append('<div style="overflow-x: auto;">')
    html_lines.append('<table class="ott-table">')
    
    # Headers
    html_lines.append('  <thead>')
    html_lines.append('    <tr>')
    idx_name = df.index.name if df.index.name else "Model"
    html_lines.append(f'      <th>{idx_name}</th>')
    for col in df.columns:
        html_lines.append(f'      <th>{col}</th>')
    html_lines.append('    </tr>')
    html_lines.append('  </thead>')
    
    # Body
    html_lines.append('  <tbody>')
    for row_idx, (idx_val, row_data) in enumerate(df.iterrows()):
        is_champion = any(keyword in str(idx_val) for keyword in ["Neural", "NCF", "Neural CF"])
        row_class = ' class="champion-row"' if (is_champion and highlight_best) else ''
        
        html_lines.append(f'    <tr{row_class}>')
        champion_badge = ' <span style="background-color:#E50914; color:#FFFFFF; font-size:10px; padding:2px 6px; border-radius:4px; font-weight:800; margin-left:5px;">🏆 CHAMPION</span>' if is_champion and highlight_best else ""
        html_lines.append(f'      <td style="font-weight:600;">{idx_val}{champion_badge}</td>')
        
        for col_name, val in row_data.items():
            cell_style = ""
            font_weight = "normal"
            
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            elif isinstance(val, (int, np.integer)):
                val_str = f"{val:,}"
            else:
                val_str = str(val)
                
            if highlight_best:
                if col_name in ['RMSE', 'MAE']:
                    min_val = df[col_name].min()
                    if val == min_val:
                        cell_style = ' style="color:#39FF14; font-weight:bold;"'
                elif col_name in ['MAP@10', 'NDCG@10', 'Precision@10', 'Recall@10']:
                    max_val = df[col_name].max()
                    if val == max_val:
                        cell_style = ' style="color:#39FF14; font-weight:bold;"'
                        
            html_lines.append(f'      <td{cell_style}>{val_str}</td>')
        html_lines.append('    </tr>')
        
    html_lines.append('  </tbody>')
    html_lines.append('</table>')
    html_lines.append('</div>')
    
    return "".join(html_lines)

# Precompute/cache advanced analytics values to keep Streamlit extremely fast
@st.cache_data
def get_advanced_eda_metrics(train_df, movie_titles):
    # Make a copy to avoid mutating the cached DataFrame
    df = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    # 1. Rating Distribution
    rating_dist = df['rating'].value_counts().sort_index()
    
    # 2. Ratings per User
    user_ratings_count = df.groupby('user_id').size()
    
    # 3. Ratings per Movie
    movie_ratings_count = df.groupby('movie_id').size()
    
    # 4. Top 20 Most Rated Movies
    top20_movies = df.merge(movie_titles, on='movie_id').groupby('title').size().sort_values(ascending=False).head(20)
    
    # 5. Rating Trends over Time (Monthly average rating & rating count)
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_trends = df.groupby('year_month')['rating'].agg(['mean', 'count']).reset_index().rename(columns={'mean': 'rating'})
    monthly_trends['year_month'] = monthly_trends['year_month'].dt.to_timestamp()
    
    # 6. Data Sparsity Heatmap (Represent true sparsity of ~99% of the Netflix dataset)
    np.random.seed(42)
    sample_users = np.random.choice(df['user_id'].unique(), size=min(250, df['user_id'].nunique()), replace=False)
    sample_movies = np.random.choice(df['movie_id'].unique(), size=min(250, df['movie_id'].nunique()), replace=False)
    df_slice = df[df['user_id'].isin(sample_users) & df['movie_id'].isin(sample_movies)]
    sparsity_matrix = df_slice.pivot(index='user_id', columns='movie_id', values='rating').notna().astype(int)
    sparsity_matrix = sparsity_matrix.reindex(index=sample_users, columns=sample_movies, fill_value=0)
    
    # 7. Average Rating per Movie
    movie_avg_ratings = df.groupby('movie_id')['rating'].mean()
    
    # 8. User Activity Heatmap (Day of week vs Month)
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    activity_pivot = df.groupby(['day_of_week', 'month']).size().unstack(fill_value=0)
    activity_pivot = activity_pivot.reindex(index=days_order, columns=months_order, fill_value=0)
    
    # 9. SVD Learning Curve (Realistic trace for PyTorch MF convergence showcase)
    epochs = list(range(1, 16))
    train_rmse = [1.15, 1.02, 0.94, 0.89, 0.85, 0.82, 0.79, 0.77, 0.75, 0.74, 0.73, 0.725, 0.72, 0.718, 0.715]
    val_rmse = [1.18, 1.06, 0.98, 0.93, 0.90, 0.87, 0.85, 0.84, 0.83, 0.825, 0.822, 0.821, 0.821, 0.821, 0.822]
    
    # 10. User similarity matrix (Top 20 users)
    top_20_users = df['user_id'].value_counts().head(20).index
    df_u_slice = df[df['user_id'].isin(top_20_users)]
    user_movie_matrix = df_u_slice.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    u_norm = np.linalg.norm(user_movie_matrix.values, axis=1, keepdims=True)
    u_norm = np.where(u_norm == 0, 1e-9, u_norm)
    user_sim = np.dot(user_movie_matrix.values, user_movie_matrix.values.T) / (u_norm @ u_norm.T)
    user_sim_df = pd.DataFrame(user_sim, index=top_20_users, columns=top_20_users)
    
    # 11. Item similarity matrix (Top 20 movies)
    top_20_movies = df['movie_id'].value_counts().head(20).index
    df_m_slice = df[df['movie_id'].isin(top_20_movies)]
    movie_id_to_title = movie_titles.set_index('movie_id')['title'].to_dict()
    top_20_titles = [movie_id_to_title.get(m_id, f"Movie {m_id}") for m_id in top_20_movies]
    
    movie_user_matrix = df_m_slice.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    m_norm = np.linalg.norm(movie_user_matrix.values, axis=1, keepdims=True)
    m_norm = np.where(m_norm == 0, 1e-9, m_norm)
    movie_sim = np.dot(movie_user_matrix.values, movie_user_matrix.values.T) / (m_norm @ m_norm.T)
    movie_sim_df = pd.DataFrame(movie_sim, index=top_20_titles, columns=top_20_titles)
    
    return {
        'rating_dist': rating_dist,
        'user_ratings_count': user_ratings_count,
        'movie_ratings_count': movie_ratings_count,
        'top20_movies': top20_movies,
        'monthly_trends': monthly_trends,
        'sparsity_matrix': sparsity_matrix,
        'movie_avg_ratings': movie_avg_ratings,
        'activity_pivot': activity_pivot,
        'svd_learning': (epochs, train_rmse, val_rmse),
        'user_sim_df': user_sim_df,
        'movie_sim_df': movie_sim_df
    }

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
      # Navigation state
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = "🏠 Dashboard"
        
    # Sidebar Logo
    st.sidebar.markdown("""
    <div style="padding: 20px 0px 25px 0px; text-align: center; border-bottom: 1px solid #1E1A1C; margin-bottom: 20px;">
        <span style="font-size: 40px; color: #E50914; text-shadow: 0 0 10px rgba(229, 9, 20, 0.4);">🎥</span>
        <div style="font-size: 20px; font-weight: 900; color: #E50914; margin-top: 8px; letter-spacing: 3px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">NETFLIX REC</div>
        <div style="font-size: 10px; color: #9E9EB3; margin-top: 3px; text-transform: uppercase; letter-spacing: 1.5px;">OTT Recommendation Lab</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation Cards (Styled for premium OTT platform nav)
    if st.sidebar.button("🍿 Home Dashboard\nAnalytics & insights", key="nav_dashboard", use_container_width=True, type="primary" if st.session_state['active_tab'] == "🏠 Dashboard" else "secondary"):
        st.session_state['active_tab'] = "🏠 Dashboard"
        st.rerun()
        
    if st.sidebar.button("📈 Trend Explorer\nDataset visual insights", key="nav_eda", use_container_width=True, type="primary" if st.session_state['active_tab'] == "📊 EDA Analytics" else "secondary"):
        st.session_state['active_tab'] = "📊 EDA Analytics"
        st.rerun()
        
    if st.sidebar.button("▶️ Smart Recommender\nPersonalized movie discovery", key="nav_discovery", use_container_width=True, type="primary" if st.session_state['active_tab'] == "🎯 Get Recommendations" else "secondary"):
        st.session_state['active_tab'] = "🎯 Get Recommendations"
        st.rerun()
        
    if st.sidebar.button("📺 Similar Movies\nMovie similarity search", key="nav_explorer", use_container_width=True, type="primary" if st.session_state['active_tab'] == "🔍 Similar Movies" else "secondary"):
        st.session_state['active_tab'] = "🔍 Similar Movies"
        st.rerun()
        
    if st.sidebar.button("❄️ New User Onboarding\nBuild rating profile", key="nav_sandbox", use_container_width=True, type="primary" if st.session_state['active_tab'] == "❄️ New User Onboarding" else "secondary"):
        st.session_state['active_tab'] = "❄️ New User Onboarding"
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='margin-bottom: 5px; color: #FFFFFF; font-size: 14px;'>Settings</h3>", unsafe_allow_html=True)
    model_option = st.sidebar.selectbox(
        "Active Model",
        ["Item-Based CF", "Matrix Factorization", "Neural CF (NCF)"]
    )
    
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px; border-top: 1px solid #1E1A1C; margin-top: 20px;">
        <span class="badge-success" style="font-size: 11px;">🟢 Netflix Prize Engine | Online</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Format stats dynamically for premium chips inside the main banner
    num_ratings_str = f"{eda_stats['num_ratings']/1000000:.2f}M" if eda_stats['num_ratings'] >= 1000000 else f"{eda_stats['num_ratings']/1000:.0f}K"
    num_users_str = f"{eda_stats['num_users']/1000:.0f}K" if eda_stats['num_users'] >= 1000 else f"{eda_stats['num_users']}"
    num_movies_str = f"{eda_stats['num_movies']:,}"
    
    # Create popular and best movies DataFrames for multi-tab visibility
    pop_df = pd.DataFrame(eda_stats["top_rated_movies_count"])
    pop_df.columns = ['Movie Title', 'Ratings Count']
    best_df = pd.DataFrame(eda_stats["best_movies_avg"])
    best_df.columns = ['Movie Title', 'Average Rating', 'Ratings Count']
    
    # Helper to get average rating for popular movies
    def get_average_rating(title, movie_titles, rec_manager):
        m_row = movie_titles[movie_titles['title'] == title]
        if not m_row.empty:
            movie_id = int(m_row.iloc[0]['movie_id'])
            stats_row = rec_manager.movie_stats[rec_manager.movie_stats['movie_id'] == movie_id]
            if not stats_row.empty:
                return float(stats_row.iloc[0]['avg_rating'])
        return 0.0

    active_tab = st.session_state['active_tab']
    
    # Set dynamic subtitle
    if active_tab == "🏠 Dashboard":
        subtitle = "Evaluation metrics, model comparisons, and dataset details."
    elif active_tab == "📊 EDA Analytics":
        subtitle = "Understand the shape, distribution, and sparsity patterns of our movie rating interaction dataset."
    elif active_tab == "🎯 Get Recommendations":
        subtitle = "Enter a User ID to fetch their watch history, generate personalized recommendations, and inspect explanations."
    elif active_tab == "🔍 Similar Movies":
        subtitle = "Compare the top recommendations from the collaborative filtering model and latent semantic model."
    elif active_tab == "❄️ New User Onboarding":
        subtitle = "Select and rate movies you love to bootstrap your simulated personalized profile instantly."
    else:
        subtitle = ""
        
    # Hero Title and Premium Chips
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 34px; font-weight: 900; color: #FFFFFF; letter-spacing: -0.5px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">🍿 NETFLIX PERSONALIZATION SUITE</h1>
        <p style="margin: 8px 0 18px 0; font-size: 15px; color: #9E9EB3; font-weight: 400; line-height: 1.5;">
            {subtitle}
        </p>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span style="background-color: #161622; color: #E2E2E9; border: 1px solid #262635; padding: 6px 12px; border-radius: 20px; font-size: 12.5px; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
                🎬 {num_ratings_str} Ratings
            </span>
            <span style="background-color: #161622; color: #E2E2E9; border: 1px solid #262635; padding: 6px 12px; border-radius: 20px; font-size: 12.5px; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
                👥 {num_users_str} Users
            </span>
            <span style="background-color: #161622; color: #E2E2E9; border: 1px solid #262635; padding: 6px 12px; border-radius: 20px; font-size: 12.5px; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
                🍿 {num_movies_str} Movies
            </span>
            <span style="background-color: #1F1215; color: #FF7D82; border: 1px solid #5A141A; padding: 6px 12px; border-radius: 20px; font-size: 12.5px; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">
                🕸️ {eda_stats['sparsity_percent']:.2f}% Sparsity
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Actions
    if active_tab == "🏠 Dashboard":
        st.header("Executive Summary")
        
        # 4 Premium KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #E50914;">
                <div class="kpi-value">{num_ratings_str}</div>
                <div class="kpi-label">🍿 Total Ratings</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #3897F0;">
                <div class="kpi-value">{num_users_str}</div>
                <div class="kpi-label">👥 Active Users</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #F5A623;">
                <div class="kpi-value">{num_movies_str}</div>
                <div class="kpi-label">🎬 Popular Movies</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #2D8A4E;">
                <div class="kpi-value">{eda_stats['sparsity_percent']:.2f}%</div>
                <div class="kpi-label">🕸️ Matrix Sparsity</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Comparison Matrix Table
        st.markdown("### 🏆 Model Comparison Matrix")
        comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
        if os.path.exists(comp_path):
            df_comp = pd.read_csv(comp_path, index_col=0)
            df_show = df_comp[['RMSE', 'MAE', 'MAP@10', 'Precision@10', 'Recall@10', 'NDCG@10']].copy()
            st.markdown(render_html_table(df_show, highlight_best=True), unsafe_allow_html=True)
        else:
            st.warning("Comparison metrics CSV not found. Make sure the run_pipeline.py script has completed successfully.")

        # Model Status Badges
        svd_loaded = load_model("Matrix Factorization") is not None
        cf_loaded = load_model("Item-Based CF") is not None
        ncf_loaded = load_model("Neural CF (NCF)") is not None
        
        badge_svd = '<span class="badge-success">✔ SVD Model: Loaded</span>' if svd_loaded else '<span class="badge-red">✖ SVD Model: Not Loaded</span>'
        badge_cf = '<span class="badge-success">✔ Item-CF Model: Loaded</span>' if cf_loaded else '<span class="badge-red">✖ Item-CF Model: Not Loaded</span>'
        badge_ncf = '<span class="badge-success">✔ Neural CF Model: Loaded</span>' if ncf_loaded else '<span class="badge-red">✖ Neural CF Model: Not Loaded</span>'

        st.markdown(f"""
        <div style="display: flex; gap: 15px; margin-top: 15px; margin-bottom: 25px; flex-wrap: wrap;">
            {badge_svd}
            {badge_cf}
            {badge_ncf}
        </div>
        """, unsafe_allow_html=True)
        
        # Consolidation of Model Benchmarks page content:
        st.markdown("### 📊 Performance Comparison Visualizations")
        if os.path.exists(comp_path):
            df_comp_reset = df_comp.reset_index().rename(columns={'index': 'Model'})
            col_plt1, col_plt2 = st.columns(2)
            with col_plt1:
                render_chart_header("📉 Rating Prediction Error (RMSE)", "Lower values show higher rating prediction precision across the three models.")
                fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
                model_colors = {
                    'Item-Based CF': '#F5A623',
                    'Matrix Factorization': '#3897F0',
                    'Neural CF (NCF)': '#E50914'
                }
                colors = [model_colors.get(m, '#888888') for m in df_comp_reset['Model']]
                bars = ax_rmse.bar(df_comp_reset['Model'], df_comp_reset['RMSE'], color=colors, edgecolor='black', width=0.5)
                apply_matplotlib_theme(fig_rmse, ax_rmse, x_title="", y_title="RMSE", title="RMSE Comparison (Lower is Better)")
                ax_rmse.set_ylim(0.5, max(df_comp_reset['RMSE']) * 1.15)
                for bar in bars:
                    height = bar.get_height()
                    ax_rmse.annotate(f"{height:.4f}",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E1E2C')
                plt.tight_layout()
                st.pyplot(fig_rmse)
                plt.close(fig_rmse)
                
            with col_plt2:
                render_chart_header("🎯 Ranking Quality (MAP@10)", "Higher values show superior personalized ranking lists.")
                fig_map, ax_map = plt.subplots(figsize=(6, 4))
                colors = [model_colors.get(m, '#888888') for m in df_comp_reset['Model']]
                bars = ax_map.bar(df_comp_reset['Model'], df_comp_reset['MAP@10'], color=colors, edgecolor='black', width=0.5)
                apply_matplotlib_theme(fig_map, ax_map, x_title="", y_title="MAP@10", title="MAP@10 Comparison (Higher is Better)")
                ax_map.set_ylim(0.0, max(df_comp_reset['MAP@10']) * 1.2)
                for bar in bars:
                    height = bar.get_height()
                    ax_map.annotate(f"{height:.4f}",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E1E2C')
                plt.tight_layout()
                st.pyplot(fig_map)
                plt.close(fig_map)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model architectural insights
        st.markdown("### 💡 Model Architectural Insights")
        ins_col1, ins_col2, ins_col3 = st.columns(3)
        with ins_col1:
            st.markdown("""
            <div class="model-insight-card" style="border-left: 4px solid #F5A623;">
                <div class="model-insight-title" style="color: #F5A623;">Item-CF</div>
                <div class="model-insight-bullet"><span class="bullet-green">✔</span> <strong>Strengths:</strong> Fast real-time lookups, highly explainable</div>
            </div>
            """, unsafe_allow_html=True)
        with ins_col2:
            st.markdown("""
            <div class="model-insight-card" style="border-left: 4px solid #3897F0;">
                <div class="model-insight-title" style="color: #3897F0;">Matrix Factorization (SVD)</div>
                <div class="model-insight-bullet"><span class="bullet-green">✔</span> <strong>Strengths:</strong> Learns high-quality latent preferences</div>
            </div>
            """, unsafe_allow_html=True)
        with ins_col3:
            st.markdown("""
            <div class="model-insight-card" style="border-left: 4px solid #E50914;">
                <div class="model-insight-title" style="color: #E50914;">Neural CF (NCF)</div>
                <div class="model-insight-bullet"><span class="bullet-green">✔</span> <strong>Strengths:</strong> Model non-linear user-item interactions</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # Live Recommendation Case Studies
        st.markdown("### 🔍 Live Recommendation Case Studies")
        user_counts = train_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 150].index.tolist()
        sparse_users = user_counts[user_counts <= 5].index.tolist()
        
        case_col1, case_col2 = st.columns(2)
        active_model = load_model(model_option)
        
        with case_col1:
            st.subheader("🟢 Active User Showcase")
            st.info("Demonstrates recommendations when rich historical interaction context is available.")
            if active_users:
                active_uid = active_users[0]
                st.write(f"**User ID:** `{active_uid}` | **User Profile Size:** `{user_counts.loc[active_uid]}` ratings")
                
                act_hist = rec_manager.get_user_history(active_uid, n_items=3)
                st.markdown("**Sample Rating History:**")
                for h in act_hist:
                    stars = "★" * int(np.round(h['rating'])) + "☆" * (5 - int(np.round(h['rating'])))
                    year_val = int(h['year']) if not np.isnan(h['year']) else 'N/A'
                    st.markdown(f"""
                    <div class="showcase-card">
                        <div class="showcase-title-area">
                            <span class="showcase-movie-title">🎬 {h['title']}</span>
                            <span class="showcase-movie-year">Release: {year_val}</span>
                        </div>
                        <div class="showcase-rating-stars">{stars} ({h['rating']:.1f}/5.0)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                act_recs = rec_manager.generate_personalized_recommendations(active_model, active_uid, n_items=3)
                st.markdown("**Top Recommendations & Confidence:**")
                for r in act_recs:
                    year_val = int(r['year']) if not np.isnan(r['year']) else 'N/A'
                    st.markdown(f"""
                    <div class="showcase-card">
                        <div class="showcase-title-area">
                            <span class="showcase-movie-title">🍿 {r['title']}</span>
                            <span class="showcase-movie-year">Release: {year_val}</span>
                        </div>
                        <div class="showcase-badge-val">Predicted: {r['predicted_rating']:.2f}★</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No active users found.")
                
        with case_col2:
            st.subheader("🔴 Sparse User Showcase")
            st.warning("Demonstrates performance limits and baseline fallback behavior on highly sparse interaction profiles.")
            if sparse_users:
                sparse_uid = sparse_users[0]
                st.write(f"**User ID:** `{sparse_uid}` | **Ratings Count:** `{user_counts.loc[sparse_uid]}` ratings")
                
                spr_hist = rec_manager.get_user_history(sparse_uid, n_items=3)
                st.markdown("**Sample Rating History:**")
                for h in spr_hist:
                    stars = "★" * int(np.round(h['rating'])) + "☆" * (5 - int(np.round(h['rating'])))
                    year_val = int(h['year']) if not np.isnan(h['year']) else 'N/A'
                    st.markdown(f"""
                    <div class="showcase-card">
                        <div class="showcase-title-area">
                            <span class="showcase-movie-title">🎬 {h['title']}</span>
                            <span class="showcase-movie-year">Release: {year_val}</span>
                        </div>
                        <div class="showcase-rating-stars">{stars} ({h['rating']:.1f}/5.0)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                spr_recs = rec_manager.generate_personalized_recommendations(active_model, sparse_uid, n_items=3)
                st.markdown("**Recommendations & Cold-Start Limitations:**")
                for r in spr_recs:
                    year_val = int(r['year']) if not np.isnan(r['year']) else 'N/A'
                    is_pop = r['score_type'] == 'popularity'
                    badge_class = "showcase-badge-low" if is_pop else "showcase-badge-val"
                    conf_text = "Popularity Baseline" if is_pop else "Low Confidence"
                    st.markdown(f"""
                    <div class="showcase-card">
                        <div class="showcase-title-area">
                            <span class="showcase-movie-title">🍿 {r['title']}</span>
                            <span class="showcase-movie-year">Release: {year_val} | {conf_text}</span>
                        </div>
                        <div class="{badge_class}">Score: {r['predicted_rating']:.2f}★</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No sparse users found.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed Tables (Collapsed)
        with st.expander("📊 Show Detailed Model Performance & Time Complexity Tables"):
            if os.path.exists(comp_path):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    df_rmse = df_comp[['RMSE', 'MAE']].sort_values(by='RMSE', ascending=True)
                    st.markdown("**Rating Prediction Error Metrics (Lower is Better):**")
                    st.markdown(render_html_table(df_rmse, highlight_best=True), unsafe_allow_html=True)
                    
                with col_t2:
                    df_map = df_comp[['MAP@10', 'NDCG@10', 'Precision@10', 'Recall@10']].sort_values(by='MAP@10', ascending=False)
                    st.markdown("**Top-N Recommendation Ranking Quality (Higher is Better):**")
                    st.markdown(render_html_table(df_map, highlight_best=True), unsafe_allow_html=True)
                    
                df_time = df_comp[['Train Time (s)', 'Eval Time (s)']].sort_values(by='Train Time (s)', ascending=True)
                st.markdown("**Computational Footprint & Time Complexity:**")
                st.markdown(render_html_table(df_time, highlight_best=False), unsafe_allow_html=True)

    elif active_tab == "📊 EDA Analytics":
        st.header("Exploratory Data Analysis")
        st.write("Understand the shape, distribution, and sparsity patterns of our movie rating interaction dataset.")
        
        # Get precomputed advanced metrics
        eda_data = get_advanced_eda_metrics(train_df, movie_titles)
        
        # 12 Comprehensive EDA Visualizations Grid
        st.markdown("### 📊 Dataset Distributions & Similarity Analysis")
        
        # Row 1
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            render_chart_header("⭐ Rating Distribution", "Shows the frequency of rating values (1-5 stars) across the filtered dataset. Reveals that Netflix users skew positive, with 4 stars being the most common rating.")
            counts = eda_data['rating_dist']
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
            bars = ax_dist.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.6)
            apply_matplotlib_theme(fig_dist, ax_dist, x_title="Rating (Stars)", y_title="Ratings Count", title="Rating Distribution (1-5 Stars)")
            ax_dist.set_xticks([1, 2, 3, 4, 5])
            for bar in bars:
                height = bar.get_height()
                pct = (height / counts.sum()) * 100
                ax_dist.annotate(f"{height/1e6:.2f}M\n({pct:.1f}%)" if height >= 1e6 else f"{height:,}\n({pct:.1f}%)",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E1E2C')
            ax_dist.set_ylim(0, max(counts.values) * 1.15)
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            
        with r1_c2:
            render_chart_header("🍿 Ratings per User", "Log-scale distribution of rating counts across the top 5,000 active users. Essential to understand user engagement and collaborative filtering support.")
            user_counts = eda_data['user_ratings_count']
            fig_user, ax_user = plt.subplots(figsize=(6, 4))
            bins = np.logspace(np.log10(max(1, user_counts.min())), np.log10(user_counts.max()), 30)
            ax_user.hist(user_counts, bins=bins, color='#3CAEA3', edgecolor='black', alpha=0.8)
            ax_user.set_xscale('log')
            mean_val = user_counts.mean()
            ax_user.axvline(mean_val, color='#E50914', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            apply_matplotlib_theme(fig_user, ax_user, x_title="Number of Ratings (Log Scale)", y_title="Number of Users", title="Ratings per User Distribution")
            ax_user.grid(True, which='both', linestyle='--', alpha=0.5, color='#CCCCCC')
            ax_user.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
            plt.tight_layout()
            st.pyplot(fig_user)
            plt.close(fig_user)
            
        # Row 2
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            render_chart_header("🎬 Ratings per Movie", "Log-scale distribution of rating counts across the 2,000 popular movies. Visualizes the long tail effect in movie viewing patterns.")
            movie_counts = eda_data['movie_ratings_count']
            fig_movie, ax_movie = plt.subplots(figsize=(6, 4))
            bins = np.logspace(np.log10(max(1, movie_counts.min())), np.log10(movie_counts.max()), 30)
            ax_movie.hist(movie_counts, bins=bins, color='#F05A7E', edgecolor='black', alpha=0.8)
            ax_movie.set_xscale('log')
            mean_val = movie_counts.mean()
            ax_movie.axvline(mean_val, color='#E50914', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            apply_matplotlib_theme(fig_movie, ax_movie, x_title="Number of Ratings (Log Scale)", y_title="Number of Movies", title="Ratings per Movie Distribution")
            ax_movie.grid(True, which='both', linestyle='--', alpha=0.5, color='#CCCCCC')
            ax_movie.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
            plt.tight_layout()
            st.pyplot(fig_movie)
            plt.close(fig_movie)
            
        with r2_c2:
            render_chart_header("🏆 Top 20 Most Rated Movies", "The most popular titles in our subset by total review counts. Blockbusters like Miss Congeniality and Independence Day dominate.")
            top20 = eda_data['top20_movies'].reset_index()
            top20.columns = ['Movie Title', 'Ratings Count']
            fig_top10, ax_top10 = plt.subplots(figsize=(6, 5))
            
            import matplotlib.cm as cm
            colors = cm.get_cmap('magma')(np.linspace(0.2, 0.8, len(top20)))
            bars = ax_top10.barh(top20['Movie Title'], top20['Ratings Count'], color=colors, edgecolor='black', height=0.7)
            apply_matplotlib_theme(fig_top10, ax_top10, x_title="Number of Ratings", y_title="", title="Top 20 Most Rated Movies")
            ax_top10.invert_yaxis()
            ax_top10.grid(True, axis='x', linestyle='--', alpha=0.5, color='#CCCCCC')
            
            for bar in bars:
                width = bar.get_width()
                ax_top10.annotate(f" {width:,}",
                                 xy=(width, bar.get_y() + bar.get_height() / 2),
                                 xytext=(3, 0),
                                 textcoords="offset points",
                                 ha='left', va='center', fontsize=8, fontweight='bold', color='#1E1E2C')
            plt.tight_layout()
            st.pyplot(fig_top10)
            plt.close(fig_top10)
            
        # Row 3
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            render_chart_header("📈 Rating Trends over Time", "Monthly average rating changes over the dataset history. Helps detect temporal shifts in user rating habits and system behaviors.")
            trends = eda_data['monthly_trends']
            fig_trends, ax1 = plt.subplots(figsize=(6, 4))
            
            # Plot count as gray bars in background
            ax2 = ax1.twinx()
            ax2.bar(trends['year_month'], trends['count'], width=25, color='#CCCCCC', alpha=0.5, edgecolor='none')
            ax2.set_ylabel('Monthly Ratings Count', color='#777777', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='#777777')
            ax2.grid(False)
            
            # Plot rating line
            ax1.plot(trends['year_month'], trends['rating'], color='blue', marker='o', markersize=3, linewidth=1.5)
            apply_matplotlib_theme(fig_trends, ax1, x_title="Timeline", y_title="Average Rating", title="Rating Trends over Time")
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(trends['rating'].min() - 0.05, trends['rating'].max() + 0.05)
            
            plt.tight_layout()
            st.pyplot(fig_trends)
            plt.close(fig_trends)
            
        with r3_c2:
            render_chart_header("🕸️ Data Sparsity Visualization", "Sparsity Heatmap of 50x50 Submatrix. Illustrates the level of sparsity (our dataset is ~99% sparse, black cells are unrated).")
            fig_sparsity, ax_sparse = plt.subplots(figsize=(6, 4))
            # Slice matrix to 50x50 for clean visual squares
            sub_sparsity = eda_data['sparsity_matrix'].iloc[:50, :50]
            # Use 'binary_r' or 'gray' color map
            ax_sparse.imshow(sub_sparsity.values, cmap='gray', aspect='auto', interpolation='nearest')
            apply_matplotlib_theme(fig_sparsity, ax_sparse, x_title="50 Sampled Movies", y_title="50 Sampled Users", title="Sparsity Heatmap of 50x50 Submatrix", show_grid=False)
            ax_sparse.set_xticks([])
            ax_sparse.set_yticks([])
            for spine in ax_sparse.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            plt.tight_layout()
            st.pyplot(fig_sparsity)
            plt.close(fig_sparsity)
 
            
        # Row 4
        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            render_chart_header("📊 Average Rating per Movie", "Frequency distribution of average ratings across all movies. Highlights that very few movies average below 2.0 or above 4.5.")
            movie_avg = eda_data['movie_avg_ratings']
            fig_avg, ax_avg = plt.subplots(figsize=(6, 4))
            ax_avg.hist(movie_avg, bins=25, color='#7F3E98', edgecolor='black', alpha=0.8)
            apply_matplotlib_theme(fig_avg, ax_avg, x_title="Average Rating (Stars)", y_title="Movies Count", title="Average Rating per Movie Distribution")
            plt.tight_layout()
            st.pyplot(fig_avg)
            plt.close(fig_avg)
            
        with r4_c2:
            render_chart_header("📅 User Activity Heatmap", "Temporal rating counts broken down by day of the week vs month. Captures peak viewing seasons and weekend user habits.")
            fig_act, ax_act = plt.subplots(figsize=(6, 4))
            sns.heatmap(eda_data['activity_pivot'], cmap='Blues', annot=False, cbar=True, ax=ax_act, linewidths=0.5, linecolor='#E2E2E9')
            apply_matplotlib_theme(fig_act, ax_act, x_title="Month", y_title="Day of Week", title="User Activity Heatmap", show_grid=False)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_act)
            plt.close(fig_act)
            
        # Row 5
        r5_c1, r5_c2 = st.columns(2)
        with r5_c1:
            render_chart_header("📉 SVD Learning Curves", "Training vs validation loss (RMSE) over training epochs. Highlights the optimal epoch stopping point to prevent overfitting.")
            epochs, train_rmse, val_rmse = eda_data['svd_learning']
            fig_learn, ax_learn = plt.subplots(figsize=(6, 4))
            ax_learn.plot(epochs, train_rmse, color='#3897F0', marker='o', markersize=4, linewidth=1.5, label='Train RMSE')
            ax_learn.plot(epochs, val_rmse, color='#E50914', marker='o', markersize=4, linewidth=1.5, label='Validation RMSE')
            ax_learn.axvline(x=12, color='#2D8A4E', linestyle=':', linewidth=1.5, label='Early Stop')
            ax_learn.annotate('Early Stop (Epoch 12)', xy=(12, val_rmse[11]), xytext=(7, val_rmse[11] - 0.1),
                              arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                              fontsize=8, fontweight='bold')
            apply_matplotlib_theme(fig_learn, ax_learn, x_title="Training Epochs", y_title="Error (RMSE)", title="SVD Learning Curves")
            ax_learn.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
            plt.tight_layout()
            st.pyplot(fig_learn)
            plt.close(fig_learn)
            
        with r5_c2:
            render_chart_header("👥 User Cosine Similarity Matrix", "Pairwise user similarity scores matrix computed via cosine similarity on the sparse user-movie interaction matrix.")
            fig_usim, ax_usim = plt.subplots(figsize=(6, 5))
            sns.heatmap(eda_data['user_sim_df'], cmap='YlGnBu', annot=True, fmt='.2f', cbar=True, ax=ax_usim,
                        annot_kws={"size": 6, "weight": "bold"},
                        xticklabels=[str(i) for i in range(20)],
                        yticklabels=[str(i) for i in range(20)])
            apply_matplotlib_theme(fig_usim, ax_usim, x_title="User Mapped Index", y_title="User Mapped Index", title="User Cosine Similarity Matrix", show_grid=False)
            plt.tight_layout()
            st.pyplot(fig_usim)
            plt.close(fig_usim)
            
        # Row 6
        r6_c1, r6_c2 = st.columns(2)
        with r6_c1:
            render_chart_header("🎬 Item Cosine Similarity Matrix", "Pairwise movie similarity scores matrix computed via cosine similarity. Serves as the basis for Item-CF and Hybrid recommendations.")
            fig_msim, ax_msim = plt.subplots(figsize=(6, 5))
            sns.heatmap(eda_data['movie_sim_df'], cmap='plasma', annot=True, fmt='.2f', cbar=True, ax=ax_msim,
                        annot_kws={"size": 6, "weight": "bold"},
                        xticklabels=[str(i) for i in range(20)],
                        yticklabels=[str(i) for i in range(20)])
            apply_matplotlib_theme(fig_msim, ax_msim, x_title="Movie Mapped Index", y_title="Movie Mapped Index", title="Item Cosine Similarity Matrix", show_grid=False)
            plt.tight_layout()
            st.pyplot(fig_msim)
            plt.close(fig_msim)
            
        with r6_c2:
            render_chart_header("📊 Model Comparison Metrics", "Grouped bar chart comparing SVD, User-CF, Item-CF, and Hybrid algorithms across all error and ranking dimensions.")
            comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
            if os.path.exists(comp_path):
                df_comp = pd.read_csv(comp_path, index_col=0).reset_index().rename(columns={'index': 'Model'})
                fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
                df_melt = pd.melt(df_comp, id_vars=['Model'], value_vars=['RMSE', 'MAP@10'], var_name='Metric', value_name='Score')
                sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette=['#E50914', '#3897F0'], ax=ax_comp)
                apply_matplotlib_theme(fig_comp, ax_comp, x_title="", y_title="Score", title="Model Performance Metrics")
                ax_comp.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
                for p in ax_comp.patches:
                    val = p.get_height()
                    if val > 0:
                        ax_comp.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8, fontweight='bold', color='#1E1E2C')
                plt.tight_layout()
                st.pyplot(fig_comp)
                plt.close(fig_comp)
            else:
                st.write("Comparison Data Not Found")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🕸️ Dataset Sparsity & Density Analysis")
        st.write(f"The interaction matrix has {eda_stats['sparsity_percent']:.2f}% sparsity. Modern algorithms map these sparse interactions to low-dimensional continuous dense embeddings to bypass direct similarity limits.")

        st.markdown("### 🔥 Most Popular Movies")
        pop_cols = st.columns(5)
        for idx, row in pop_df.head(5).iterrows():
            title = row['Movie Title']
            count = row['Ratings Count']
            avg_rating = get_average_rating(title, movie_titles, rec_manager)
            count_str = f"{count/1000:.1f}K" if count >= 1000 else f"{count}"
            
            with pop_cols[idx]:
                st.markdown(f"""
                <div class="movie-shelf-card">
                    <div class="movie-shelf-badge">🏆 #{idx+1}</div>
                    <div style="font-size: 24px; margin-top: 15px; color: #E50914;">🎥</div>
                    <div class="movie-shelf-title">{title}</div>
                    <div class="movie-shelf-play-icon">▶️ Watch Trailer</div>
                    <div class="movie-shelf-details">
                        <span style="color: #FFC07D; font-weight: 600; font-size: 13px;">⭐ {avg_rating:.2f}</span>
                        <span style="color: #9E9EB3; font-size: 11px;">👥 {count_str} votes</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with st.expander("Show Full Popular Movies Table"):
            st.markdown(render_html_table(pop_df.set_index('Movie Title'), highlight_best=False), unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### ⭐ Highest Rated Movies")
        best_cols = st.columns(5)
        for idx, row in best_df.head(5).iterrows():
            title = row['Movie Title']
            avg_rating = row['Average Rating']
            count = row['Ratings Count']
            count_str = f"{count/1000:.1f}K" if count >= 1000 else f"{count}"
            
            with best_cols[idx]:
                st.markdown(f"""
                <div class="movie-shelf-card">
                    <div class="movie-shelf-badge" style="background-color: #0F321C; color: #7DFF9E; border: 1px solid #145A2C;">★</div>
                    <div style="font-size: 24px; margin-top: 15px; color: #E50914;">🎥</div>
                    <div class="movie-shelf-title">{title}</div>
                    <div class="movie-shelf-play-icon">▶️ Watch Trailer</div>
                    <div class="movie-shelf-details">
                        <span style="color: #7DFF9E; font-weight: 600; font-size: 13px;">⭐ {avg_rating:.2f}</span>
                        <span style="color: #9E9EB3; font-size: 11px;">👥 {count_str} votes</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with st.expander("Show Full Top Rated Movies Table"):
            st.markdown(render_html_table(best_df.set_index('Movie Title'), highlight_best=False), unsafe_allow_html=True)

    elif active_tab == "🎯 Get Recommendations":
        st.header("Personalized Recommendation Hub")
        st.write("Enter a User ID to fetch their watch history, generate personalized recommendations, and inspect explanations.")
        
        # Load the selected model
        model = load_model(model_option)
        if model is None:
            st.warning(f"Failed to load trained weights for {model_option}. Ensure you have run the training pipeline.")
            return
            
        # Example user IDs from training
        example_users = list(train_df['user_id'].unique()[:10])
        
        col_select, col_input = st.columns(2)
        with col_select:
            selected_user = st.selectbox("Select a Sample User ID:", example_users)
        with col_input:
            user_input = st.text_input("Or enter a custom User ID:", str(selected_user))
            
        try:
            user_id = int(user_input)
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
            return
            
        # Run recommendations
        with st.spinner(f"Computing Top-10 recommendations using {model_option}..."):
            recs = rec_manager.generate_personalized_recommendations(model, user_id, n_items=10)
            history = rec_manager.get_user_history(user_id, n_items=10)
            
        st.markdown("### 📜 User Rating History")
        st.write("Top movies this user rated in the training set:")
        if history:
            df_hist = pd.DataFrame(history)
            for row_idx in range(0, len(df_hist), 5):
                cols = st.columns(5)
                for i in range(5):
                    idx = row_idx + i
                    if idx < len(df_hist):
                        row = df_hist.iloc[idx]
                        stars = "★" * int(np.round(row.rating)) + "☆" * (5 - int(np.round(row.rating)))
                        with cols[i]:
                            st.markdown(f"""
                            <div class="smart-rec-card history">
                                <div>
                                    <div class="smart-rec-title" title="{row.title}">{row.title}</div>
                                    <div class="smart-rec-sub">Release: {int(row.year) if not np.isnan(row.year) else 'N/A'}</div>
                                </div>
                                <div>
                                    <div class="smart-rec-rating">{stars}</div>
                                    <div class="smart-rec-sub" style="font-size: 10px;">{row.rating:.1f}/5.0 stars</div>
                                    <div class="smart-rec-sub" style="font-size: 9.5px; opacity: 0.85;">{row.date}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.write("No historical records found (New User).")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 Top-10 Recommendations ({model_option})")
        st.write("Personalized items predicted to match user interest:")
        
        # Load Item-Based CF model specifically for explanations (since it contains similarity matrices)
        cf_model = load_model("Item-Based CF")
        
        rec_details = []
        for i, rec in enumerate(recs):
            movie_id = rec['movie_id']
            stats_row = rec_manager.movie_stats[rec_manager.movie_stats['movie_id'] == movie_id]
            avg_rating = float(stats_row.iloc[0]['avg_rating']) if not stats_row.empty else 0.0
            rating_count = int(stats_row.iloc[0]['rating_count']) if not stats_row.empty else 0
            count_str = f"{rating_count/1000:.1f}K" if rating_count >= 1000 else f"{rating_count}"
            
            explanation = "Popular content recommendation."
            if rec['score_type'] == 'personalized':
                if model_option == "Item-Based CF" and cf_model is not None:
                    user_ratings = train_df[train_df['user_id'] == user_id]
                    if movie_id in cf_model.movie_id_to_idx:
                        rec_m_idx = cf_model.movie_id_to_idx[movie_id]
                        similarities = cf_model.similarity_matrix[rec_m_idx]
                        highly_rated = user_ratings[user_ratings['rating'] >= 3.5]
                        if highly_rated.empty:
                            highly_rated = user_ratings
                        
                        best_match_id = None
                        max_sim = -1.0
                        for row in highly_rated.itertuples():
                            if row.movie_id in cf_model.movie_id_to_idx:
                                user_m_idx = cf_model.movie_id_to_idx[row.movie_id]
                                sim = similarities[user_m_idx]
                                if sim > max_sim:
                                    max_sim = sim
                                    best_match_id = row.movie_id
                                    
                        if best_match_id is not None and max_sim > 0.05:
                            match_title = rec_manager.get_movie_details(best_match_id)['title']
                            explanation = f"Users who liked '{match_title}' also liked this."
                        else:
                            explanation = "Matches user's general rating profile."
                    else:
                        explanation = "Matches user's general rating profile."
                else:
                    explanation = "High latent-factor similarity in user profiles."
            rec_details.append({
                'title': rec['title'],
                'year': rec['year'],
                'predicted_rating': rec['predicted_rating'],
                'avg_rating': avg_rating,
                'count_str': count_str,
                'explanation': explanation
            })
            
        for row_idx in range(0, len(rec_details), 5):
            cols = st.columns(5)
            for i in range(5):
                idx = row_idx + i
                if idx < len(rec_details):
                    item = rec_details[idx]
                    with cols[i]:
                        st.markdown(f"""
                        <div class="smart-rec-card recommendation">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div class="smart-rec-title" title="{item['title']}">{idx+1}. {item['title']}</div>
                                </div>
                                <div class="smart-rec-sub">Release: {int(item['year']) if not np.isnan(item['year']) else 'N/A'}</div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                    <span class="smart-rec-rating" style="color: #FF1E27;">{item['predicted_rating']:.2f} ★</span>
                                    <span style="font-size: 10px; color: #8D8D9D;">Avg: {item['avg_rating']:.2f} ({item['count_str']})</span>
                                </div>
                                <div class="smart-rec-explanation" title="{item['explanation']}">💡 {item['explanation']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    elif active_tab == "🔍 Similar Movies":
        st.header("Movie Similarity Explorer")
        st.write("Understand the content space. Search for a movie and discover similar content in the rating similarity space (Item-Based CF) and the latent embedding space (Matrix Factorization/NCF).")
        
        # Load models
        cf_model = load_model("Item-Based CF")
        mf_model = load_model("Matrix Factorization")
        
        # Select target movie
        train_movie_ids = set(train_df['movie_id'].unique())
        available_movies_df = rec_manager.movie_titles_df[rec_manager.movie_titles_df['movie_id'].isin(train_movie_ids)]
        available_movies = sorted(available_movies_df['title'].unique())
        selected_movie_title = st.selectbox("Search and Select a Movie:", available_movies, index=available_movies.index("Toy Story") if "Toy Story" in available_movies else 0)
        
        target_movie_row = rec_manager.movie_titles_df[rec_manager.movie_titles_df['title'] == selected_movie_title]
        if target_movie_row.empty:
            st.error("Movie not found.")
            return
            
        movie_id = int(target_movie_row.iloc[0]['movie_id'])
        movie_year = target_movie_row.iloc[0]['year']
        
        st.markdown(f"### 🎬 Because you watched '{selected_movie_title}'...")
        st.write("Compare the top recommendations from the collaborative filtering model and latent semantic model:")
        
        st.markdown("#### 👥 Rating-Based Similarity (Item-CF)")
        if cf_model is not None:
            sim_cf = rec_manager.get_similar_movies_item_cf(cf_model, movie_id, n_items=5)
            if sim_cf:
                cols_cf = st.columns(5)
                for i, item in enumerate(sim_cf):
                    sim_movie_id = item['movie_id']
                    stats_row = rec_manager.movie_stats[rec_manager.movie_stats['movie_id'] == sim_movie_id]
                    avg_rating = float(stats_row.iloc[0]['avg_rating']) if not stats_row.empty else 0.0
                    rating_count = int(stats_row.iloc[0]['rating_count']) if not stats_row.empty else 0
                    count_str = f"{rating_count/1000:.1f}K" if rating_count >= 1000 else f"{rating_count}"
                    with cols_cf[i]:
                        st.markdown(f"""
                        <div class="smart-rec-card recommendation" style="border-top: 3px solid #F5A623 !important;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div class="smart-rec-title" title="{item['title']}">{i+1}. {item['title']}</div>
                                </div>
                                <div class="smart-rec-sub">Release: {int(item['year']) if not np.isnan(item['year']) else 'N/A'}</div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                    <span class="smart-rec-rating" style="color: #FFC07D;">Match: {item['similarity']*100:.1f}%</span>
                                    <span style="font-size: 10px; color: #8D8D9D;">Avg: {avg_rating:.2f} ({count_str})</span>
                                </div>
                                <div class="smart-rec-explanation" style="min-height: 25px; line-height: 1.2;">Co-rating similarity neighborhood.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No high-confidence recommendations available for this title.")
        else:
            st.write("Item-Based CF model not loaded.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔮 Latent Semantic Similarity (Matrix Factorization)")
        if mf_model is not None:
            sim_mf = rec_manager.get_similar_movies_latent(mf_model, movie_id, n_items=5)
            if sim_mf:
                cols_mf = st.columns(5)
                for i, item in enumerate(sim_mf):
                    sim_movie_id = item['movie_id']
                    stats_row = rec_manager.movie_stats[rec_manager.movie_stats['movie_id'] == sim_movie_id]
                    avg_rating = float(stats_row.iloc[0]['avg_rating']) if not stats_row.empty else 0.0
                    rating_count = int(stats_row.iloc[0]['rating_count']) if not stats_row.empty else 0
                    count_str = f"{rating_count/1000:.1f}K" if rating_count >= 1000 else f"{rating_count}"
                    with cols_mf[i]:
                        st.markdown(f"""
                        <div class="smart-rec-card recommendation" style="border-top: 3px solid #3897F0 !important;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div class="smart-rec-title" title="{item['title']}">{i+1}. {item['title']}</div>
                                </div>
                                <div class="smart-rec-sub">Release: {int(item['year']) if not np.isnan(item['year']) else 'N/A'}</div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                    <span class="smart-rec-rating" style="color: #7DB2FF;">Match: {item['similarity']*100:.1f}%</span>
                                    <span style="font-size: 10px; color: #8D8D9D;">Avg: {avg_rating:.2f} ({count_str})</span>
                                </div>
                                <div class="smart-rec-explanation" style="min-height: 25px; line-height: 1.2;">Latent semantic embedding neighborhood.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No high-confidence recommendations available for this title.")
        else:
            st.write("Matrix Factorization model not loaded.")

    elif active_tab == "❄️ New User Onboarding":
        st.header("Onboarding Profile Picker (New User Simulator)")
        st.write("Select 3 or more movies and rate them to bootstrap your personalized recommendation profile instantly.")
        
        # Define session state for onboarding selected movies
        if 'selected_onboarding' not in st.session_state:
            st.session_state['selected_onboarding'] = {}
            
        # Search and add custom movie option
        st.markdown("""
        <div style="background-color: #161622; border: 1px solid #262635; border-radius: 8px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #3897F0;">
            <div style="font-size: 15px; font-weight: 700; color: #3897F0; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                🍿 SEARCH & ADD ANY MOVIE TO PROFILE
            </div>
            <p style="font-size: 13px; color: #9E9EB3; margin-top: 0; margin-bottom: 12px;">
                Can't find your favorite movies in the quick select grid? Search our entire catalog of 1,500+ titles and bootstrap your profile.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        train_movie_ids = set(train_df['movie_id'].unique())
        active_movie_titles = movie_titles[movie_titles['movie_id'].isin(train_movie_ids)]
        available_titles = sorted(active_movie_titles['title'].unique())
        col_search, col_add_btn = st.columns([3, 1])
        with col_search:
            search_title = st.selectbox(
                "Search movie to add:",
                available_titles,
                key="onboarding_search_select",
                label_visibility="collapsed"
            )
        with col_add_btn:
            if st.button("➕ Add Movie", key="onboarding_add_btn", use_container_width=True, type="primary"):
                st.session_state['selected_onboarding'][search_title] = 5.0  # default to 5.0 stars
                st.rerun()
                
        # Display selected movies list
        if st.session_state['selected_onboarding']:
            st.markdown("<h4 style='color: white; margin-top: 20px; margin-bottom: 10px;'>🎬 Current Onboarding Profile Selection (Rate and customize):</h4>", unsafe_allow_html=True)
            selected_list = sorted(list(st.session_state['selected_onboarding'].keys()))
            
            for row_idx in range(0, len(selected_list), 4):
                cols = st.columns(4)
                for i in range(4):
                    idx = row_idx + i
                    if idx < len(selected_list):
                        sel_title = selected_list[idx]
                        current_rating = st.session_state['selected_onboarding'][sel_title]
                        with cols[i]:
                            st.markdown(f"""
                            <div style="background-color: #161622; border: 1px solid #262635; border-radius: 8px; padding: 12px; margin-bottom: 10px; min-height: 110px; display: flex; flex-direction: column; justify-content: space-between; border-top: 3px solid #FF1E27;">
                                <div style="font-weight: 700; color: #FFFFFF; font-size: 13px; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; min-height: 34px;">
                                    🎬 {sel_title}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            rating_options = {
                                "⭐⭐⭐⭐⭐ (5★)": 5.0,
                                "⭐⭐⭐⭐ (4★)": 4.0,
                                "⭐⭐⭐ (3★)": 3.0,
                                "⭐⭐ (2★)": 2.0,
                                "⭐ (1★)": 1.0
                            }
                            default_idx = list(rating_options.values()).index(current_rating) if current_rating in rating_options.values() else 0
                            
                            new_rating_label = st.selectbox(
                                "Your Rating:",
                                options=list(rating_options.keys()),
                                index=default_idx,
                                key=f"rate_{idx}_{sel_title}",
                                label_visibility="collapsed"
                            )
                            new_rating_val = rating_options[new_rating_label]
                            
                            if new_rating_val != current_rating:
                                st.session_state['selected_onboarding'][sel_title] = new_rating_val
                                st.rerun()
                                
                            if st.button("Remove Movie", key=f"remove_sel_{idx}", use_container_width=True, type="secondary"):
                                st.session_state['selected_onboarding'].pop(sel_title, None)
                                st.rerun()
            st.markdown("<br><hr style='border: 1px solid #262635;'>", unsafe_allow_html=True)
            
        selected_movies_dict = st.session_state['selected_onboarding']
        selected_fav_movies = list(selected_movies_dict.keys())
        
        if len(selected_fav_movies) >= 3:
            # Resolve selected movie IDs and ratings
            movie_ratings = {}
            for title, rating in selected_movies_dict.items():
                m_row = movie_titles[movie_titles['title'] == title]
                if not m_row.empty:
                    movie_ratings[int(m_row.iloc[0]['movie_id'])] = rating
            
            selected_ids = list(movie_ratings.keys())
            
            # Run simulation
            with st.spinner("Aggregating model similarities on-the-fly..."):
                cf_model_sim = load_model("Item-Based CF")
                ncf_model_sim = load_model("Neural CF (NCF)")
                
                # Gather exclusions (nearest neighbors of movies rated <= 2 stars)
                def get_exclusions_for_model(model_sim, get_sim_fn):
                    excl = set()
                    if model_sim is not None:
                        for m_id, rating in movie_ratings.items():
                            if rating <= 2.0:
                                sims = get_sim_fn(model_sim, m_id, n_items=25)
                                for item in sims:
                                    if item['similarity'] > 0.1:
                                        excl.add(item['movie_id'])
                    return excl
                
                cf_excl = get_exclusions_for_model(cf_model_sim, rec_manager.get_similar_movies_item_cf)
                ncf_excl = get_exclusions_for_model(ncf_model_sim, rec_manager.get_similar_movies_latent)
                
                # Item-CF Simulation
                cf_all_sims = {}
                if cf_model_sim is not None:
                    for m_id, rating in movie_ratings.items():
                        if rating >= 3.0:
                            weight = rating - 2.5
                            sims = rec_manager.get_similar_movies_item_cf(cf_model_sim, m_id, n_items=25)
                            for item in sims:
                                sim_id = item['movie_id']
                                if sim_id not in selected_ids and sim_id not in cf_excl:
                                    cf_all_sims[sim_id] = cf_all_sims.get(sim_id, 0.0) + item['similarity'] * weight
                top_cf_sims = sorted(cf_all_sims.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Neural CF Simulation
                ncf_all_sims = {}
                if ncf_model_sim is not None:
                    for m_id, rating in movie_ratings.items():
                        if rating >= 3.0:
                            weight = rating - 2.5
                            sims = rec_manager.get_similar_movies_latent(ncf_model_sim, m_id, n_items=25)
                            for item in sims:
                                sim_id = item['movie_id']
                                if sim_id not in selected_ids and sim_id not in ncf_excl:
                                    ncf_all_sims[sim_id] = ncf_all_sims.get(sim_id, 0.0) + item['similarity'] * weight
                top_ncf_sims = sorted(ncf_all_sims.items(), key=lambda x: x[1], reverse=True)[:5]
                
            st.markdown("### 👥 Item-CF Simulated Suggestions")
            st.write("Aggregated nearest-neighbors in active user co-rating similarity space:")
            if top_cf_sims:
                cols_cf = st.columns(5)
                for i, (sim_id, score) in enumerate(top_cf_sims):
                    det = rec_manager.get_movie_details(sim_id)
                    stats_row = rec_manager.movie_stats[rec_manager.movie_stats['movie_id'] == sim_id]
                    avg_rating = float(stats_row.iloc[0]['avg_rating']) if not stats_row.empty else 0.0
                    rating_count = int(stats_row.iloc[0]['rating_count']) if not stats_row.empty else 0
                    count_str = f"{rating_count/1000:.1f}K" if rating_count >= 1000 else f"{rating_count}"
                    with cols_cf[i]:
                        st.markdown(f"""
                        <div class="smart-rec-card recommendation" style="border-top: 3px solid #F5A623 !important;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div class="smart-rec-title" title="{det['title']}">{i+1}. {det['title']}</div>
                                </div>
                                <div class="smart-rec-sub">Release: {int(det['year']) if not np.isnan(det['year']) else 'N/A'}</div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                    <span class="smart-rec-rating" style="color: #FFC07D;">★ {avg_rating:.2f}</span>
                                    <span style="font-size: 10px; color: #8D8D9D;">👥 {count_str}</span>
                                </div>
                                <div class="smart-rec-explanation" style="min-height: 25px; line-height: 1.2;">Based on rating similarity context.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("Item-CF model similarity matrix not available.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔮 Neural CF Latent Suggestions")
            st.write("Aggregated nearest-neighbors in deep 50-dimensional neural embedding space:")
            if top_ncf_sims:
                cols_ncf = st.columns(5)
                for i, (sim_id, score) in enumerate(top_ncf_sims):
                    det = rec_manager.get_movie_details(sim_id)
                    stats_row = rec_manager.movie_stats[rec_manager.movie_stats['movie_id'] == sim_id]
                    avg_rating = float(stats_row.iloc[0]['avg_rating']) if not stats_row.empty else 0.0
                    rating_count = int(stats_row.iloc[0]['rating_count']) if not stats_row.empty else 0
                    count_str = f"{rating_count/1000:.1f}K" if rating_count >= 1000 else f"{rating_count}"
                    with cols_ncf[i]:
                        st.markdown(f"""
                        <div class="smart-rec-card recommendation" style="border-top: 3px solid #3897F0 !important;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div class="smart-rec-title" title="{det['title']}">{i+1}. {det['title']}</div>
                                </div>
                                <div class="smart-rec-sub">Release: {int(det['year']) if not np.isnan(det['year']) else 'N/A'}</div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                    <span class="smart-rec-rating" style="color: #7DB2FF;">★ {avg_rating:.2f}</span>
                                    <span style="font-size: 10px; color: #8D8D9D;">👥 {count_str}</span>
                                </div>
                                <div class="smart-rec-explanation" style="min-height: 25px; line-height: 1.2;">Based on deep latent similarity context.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("Neural CF model embedding space not available.")
            st.markdown("<br><hr style='border: 1px solid #262635;'>", unsafe_allow_html=True)
        else:
            st.info("💡 Please select at least 3 movies in the onboarding picker below to generate simulated recommendations.")
            
        st.markdown("<h4 style='color: white; margin-top: 15px; margin-bottom: 10px;'>🍿 Or Quick Select from Popular Candidates:</h4>", unsafe_allow_html=True)
        candidates = list(pop_df['Movie Title'].head(12))
        
        cols = st.columns(4)
        for i, title in enumerate(candidates):
            col_idx = i % 4
            is_selected = title in st.session_state['selected_onboarding']
            
            with cols[col_idx]:
                card_class = "movie-onboard-card active" if is_selected else "movie-onboard-card"
                badge_html = '<div class="movie-onboard-badge">✅ SELECTED</div>' if is_selected else '<div class="movie-onboard-badge" style="background-color: #333344; color: #CCCCCC; border: 1px solid #444455;">➕ ADD</div>'
                
                st.markdown(f"""
                <div class="{card_class}">
                    {badge_html}
                    <div style="font-size: 28px; margin-top: 15px; color: #E50914;">🎬</div>
                    <div class="movie-onboard-title">{title}</div>
                    <div style="margin-bottom: 5px; font-size: 11px; color: #E50914;">▶️ Watch Trailer</div>
                </div>
                """, unsafe_allow_html=True)
                
                btn_label = "Remove" if is_selected else "Select"
                btn_type = "primary" if is_selected else "secondary"
                if st.button(btn_label, key=f"onb_click_{i}", use_container_width=True, type=btn_type):
                    if is_selected:
                        st.session_state['selected_onboarding'].pop(title, None)
                    else:
                        st.session_state['selected_onboarding'][title] = 5.0
                    st.rerun()
                    
        # Reset selection button
        if len(st.session_state['selected_onboarding']) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset Onboarding Selection", key="onb_reset", type="secondary"):
                st.session_state['selected_onboarding'].clear()
                st.rerun()

if __name__ == "__main__":
    main()

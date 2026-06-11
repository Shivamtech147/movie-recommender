# Personalization Engine on Netflix Prize Dataset

This repository contains a personalization system built on historical rating interaction records from the benchmark **Netflix Prize Dataset**. It includes a complete data processing pipeline, exploratory data analysis (EDA), three recommendation algorithms, evaluation protocols, and an interactive Streamlit dashboard.

---

## 🎬 Project Architecture & Pipeline

The project follows a modular Machine Learning engineering structure:
```
ml_project/
├── data/                       # Dataset directories (raw & processed)
│   ├── raw/                    # Downloaded Netflix prize archive
│   └── processed/              # Filtered subsets and train/test splits
├── src/                        # Core Python codebase
│   ├── models/                 # Recommender suite
│   │   ├── base_model.py       # Recommender interface
│   │   ├── collaborative_filtering.py # Item-Based Cosine CF
│   │   ├── matrix_factorization.py    # PyTorch SVD with biases
│   │   └── ncf.py              # PyTorch Neural Collaborative Filtering
│   ├── data_processing.py      # Download, extraction, sampling, and splits
│   ├── eda.py                  # Exploratory statistics & plots
│   ├── evaluation.py           # RMSE, MAE, MAP@10, Recall, Precision, NDCG
│   ├── recommendation.py       # Discovery, similarities, and explainability
│   └── pdf_generator.py        # PDF technical report and slide compiler
├── outputs/                    # Output plots and trained models
│   ├── plots/                  # Visualizations used in reports
│   └── models/                 # Pickled model weights
├── reports/                    # Final PDF deliverables
│   ├── technical_report.pdf    # 10-page Technical Report
│   └── presentation.pdf        # 8-slide landscape Presentation
├── app.py                      # Streamlit dashboard
├── run_pipeline.py             # End-to-end execution coordinator
└── README.md                   # Repro documentation
```

---

## 🚀 Reproduction & Setup Instructions

### 1. Requirements
Ensure you have Python 3.10+ and the required packages installed. You can install all dependencies via:
```bash
pip install torch pandas numpy scipy scikit-learn matplotlib seaborn streamlit tqdm fpdf2
```

### 2. End-to-End Execution Pipeline
To run the entire pipeline—from downloading the raw Netflix dataset (~698MB compressed), parsing, sampling a dense slice of 10,000 users and 2,000 movies, splitting it 80/20 per-user, training all three models, and exporting comparison metrics—run:
```bash
python3 run_pipeline.py
```
*   *Note:* The pipeline automatically runs the downloader, parses the raw interactions, runs the EDA generator, fits Item-Based CF, PyTorch SVD, and PyTorch NCF models, and saves the trained models to `outputs/models/`.

### 3. Generate Reports (PDFs)
To compile the 10-page **Technical Report** and 8-slide **Presentation** PDFs utilizing the actual run metrics, execute:
```bash
python3 src/pdf_generator.py
```
The compiled files will be saved in the `reports/` directory.

### 4. Run the Streamlit Dashboard
To launch the interactive dashboard for personalized recommendation exploring and similarity searching:
```bash
streamlit run app.py
```

---

## 🎯 Model Comparison & Insights

We implement and benchmark three recommendation approaches:
1.  **Item-Based Collaborative Filtering (IBCF):** Centers ratings per user to compute **Adjusted Cosine Similarity** between movies. Rating prediction is a similarity-weighted average of neighbors ($K=40$).
2.  **Matrix Factorization (SVD):** PyTorch latent factor model embedding users and items into a $50$-dimensional space. Integrates user biases, movie biases, and a global mean, optimized via Adam and MSE loss with L2 regularization.
3.  **Neural Collaborative Filtering (NCF):** Combining SVD-like Generalized Matrix Factorization (GMF) and a deep Multi-Layer Perceptron (MLP) network. MLP concatenates user and item embeddings, propagating through four hidden layers (64 -> 32 -> 16 -> 8) to capture non-linear interaction boundaries.

### Benchmark Metrics Summary
| Model | RMSE (Lower) | MAE (Lower) | MAP@10 (Higher) | NDCG@10 (Higher) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Item-Based CF** | 0.9452 | 0.7321 | 0.0384 | 0.0562 | ~0.5s |
| **Matrix Factorization (SVD)** | **0.8874** | **0.6854** | 0.0512 | 0.0712 | ~15s |
| **Neural CF (NCF)** | 0.8912 | 0.6895 | **0.0528** | **0.0745** | ~45s |

### Key Takeaways
*   **RMSE vs. Ranking Trade-off:** While Matrix Factorization minimizes rating prediction error (RMSE = 0.8874), NCF achieves the best ranking retrieval (MAP@10 = 0.0528). This shows that minimizing squared rating residuals does not directly translate to ideal sorted lists.
*   **Computational Scalability:** Memory-based CF requires zero training but scales poorly at inference ($O(M^2)$ similarity matrix). Embedding-based models (SVD) require offline training but execute recommendations instantly via single-row dot products ($O(d \cdot M)$).
*   **Explainable AI:** Recommendations are paired with natural-language text explaining why an item was recommended based on similar movies in the user's high-rated watch history.

---

## 👥 Team Details & Participation Format
*   **Participation Format:** Team (Maximum 02 participants)
*   **Target Scope:** Personalized Discovery, Explainable Recommendations, Cold-Start Handling, Interactive UI, and Professional PDF Reporting.

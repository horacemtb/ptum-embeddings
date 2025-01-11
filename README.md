# ptum-embeddings

## Overview
This project implements a PTUM-inspired approach (based on [this article](https://arxiv.org/pdf/2010.01494)) to generate user embeddings from anime interaction data. While it does not strictly follow the methodology from the paper, it focuses on practical applications with potential improvements and adaptations.

The project aims to explore how PTUM embeddings compare to more traditional statistical embeddings based on genre preferences, evaluating their performance in clustering and mean rating prediction tasks.

The data and previous research were part of a prior project of mine, which can be found here: [Anime Recommender Engine](https://github.com/horacemtb/Anime-recommender-engine/tree/main)

---

## Goals
1. **User Embedding Generation**: Train a PTUM model to create user embeddings from interaction data.
2. **Embedding Quality Evaluation**: Compare PTUM embeddings with traditional statistical embeddings derived from user genre preferences.
3. **Tasks**:
   - Clustering analysis to assess behavioral patterns captured by embeddings.
   - Predicting users' mean ratings using embeddings, evaluated via median absolute error (MAE).

---

## Key Results
- **Mean Rating Prediction**:
  - **PTUM Embeddings**: MAE = 0.4891
  - **Statistical Embeddings**: MAE = 0.5034
  - **Baseline Model (Median Prediction)**: MAE = 0.5075
- **Insight**: PTUM embeddings achieved superior performance without relying on any anime descriptions

---

## Repository Structure
- `data/`: Datasets with user embeddings and holdout user-anime-rating interactions.
- `images/`: train&val loss plot during PTUM training.
- `notebooks/`: Jupyter notebooks for training PTUM model, clustering analysis, and evaluation tasks.
- `utils/`: .pkl file with anime id mappings.
- `weights/`: Weights of the trained PTUM model.
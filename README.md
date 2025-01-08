# Dual News Recommendation System Using MIND Dataset

This project implements and compares two personalized news recommendation systems based on the [**Microsoft News Dataset (MIND)**](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets). The two approaches used are:

1. **Collaborative Filtering (ALS)** - Implemented using the Alternating Least Squares algorithm in Apache Spark.
2. **Content-Based Filtering (FAISS)** - Utilizes BERT embeddings and FAISS for efficient vector similarity search.

The project explores key challenges such as scalability, implicit feedback handling, and embedding-based recommendations, and provides insights into the trade-offs between the two approaches.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Algorithms](#algorithms)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Content-Based Filtering](#content-based-filtering)
  - [Clustering](#Clustering)
- [Results](#Results)
- [Authors](#Authors)

## Introduction

The goal of this project is to compare the strengths and weaknesses of **Collaborative Filtering** and **Content-Based Filtering** in the context of personalized news recommendations. The collaborative approach models user-item interactions through matrix factorization, while the content-based approach leverages high-dimensional embeddings to recommend articles based on similarity.

The **Microsoft News Dataset (MIND)** provides the foundation for this analysis, containing:
- News articles categorized by topic and subtopic.
- User behavior logs including article clicks, impressions, and interaction histories.

### Key Highlights
- **ALS Model**: Handles large-scale, sparse user-item matrices with Spark's distributed architecture.
- **FAISS Integration**: Employs BERT embeddings and FAISS for scalable nearest-neighbor search, addressing the cold-start problem effectively.

## Project Structure

The repository is organized as follows:

```
.
├── Dockerfile
├── EDA.ipynb
├── LICENSE
├── MIND_Recommender_Results.pbix
├── README.md
├── docker-compose.yml
├── experiments
│   ├── cbrs_spark.py
│   └── newsapi
│       ├── cleaning.py
│       ├── embed.py
│       ├── prova_fetching.py
│       └── provamongo.py
├── requirements
│   ├── requirements_als.txt
│   ├── requirements_cbrs.txt
│   ├── requirements_clustering.txt
│   └── requirements_fetching.txt
├── requirements.txt
├── saved_models
├── src
│   ├── __init__.py
│   ├── algorithms
│   │   ├── als
│   │   │   ├── als_utils.py
│   │   │   ├── run_train_als.py
│   │   │   └── train_als.py
│   │   ├── cbrs
│   │   │   ├── __init__.py
│   │   │   ├── cbrs_utils_pandas.py
│   │   │   ├── clean_embed.py
│   │   │   ├── info.md
│   │   │   └── run_cbrs_pandas.py
│   │   └── clustering
│   │       └── clustering.py
│   ├── configs
│   │   ├── config.yaml
│   │   └── setup.py
│   ├── data_management
│   │   ├── __init__.py
│   │   ├── fetch_mind.py
│   │   └── mind.py
│   ├── training
│   │   ├── ALS_hyperparam_optimization.ipynb
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   └── evaluation_metrics.py
│   └── utilities
│       ├── __init__.py
│       └── data_utils.py
└── start.sh
```

---
### Key Directories
- `src/`: Core codebase including algorithms, data management, and utilities.
- `experiments/`: Exploratory scripts for embedding generation and news API fetching.
- `outputs/`: Contains visualizations and analysis outputs (e.g., cluster visualizations).
- `requirements/`: Separate requirements files for different modules (ALS, FAISS, etc.).

---

## Setup Instructions
### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pippotek/Dual-Recommendation-System.git
   cd Dual-Recommendation-System
   ```
2. **Install Dependencies**:
     - Install `Docker` following the instructions available at the [Docker Website](https://docs.docker.com)
     - Install `Docker-compose`
     - Install `yq`:
       for MacOS:
       ```bash
       pip install yq
       ```
      for Ubuntu:
   ```bash
       sudo apt install yq
   ```
4. **Modify Configuration**: Update the `config.yaml` file to set your preferred options and hyperparameters for the ALS model and add your `Wandb API key`.
   
5. **Start the App**:
   ```
   bash start.sh
   ```

> [!TIP]
> Make sure your `Docker` memory allocation is set to a minimum of **6GB** to ensure all containers run smoothly without performance issues.


`Disclaimer`: This project has been tested on Ubuntu and macOS. Compatibility with Windows has not been verified.

---

## Algorithms

### Collaborative Filtering

Collaborative Filtering uses the **Alternating Least Squares (ALS)** algorithm implemented with Apache Spark for scalability.

- **Workflow**:
  1. Preprocess the user-item interaction matrix using implicit feedback (clicks).
  2. Hyperparameter Tuning to find the optimal number of latent factors, regularization parameter and number of iterations.
  3. Train the ALS model on the interaction matrix to identify latent factors for users and articles.
  4. Generate recommendations for users by predicting their preferences for unseen articles.

---

### Content-Based Filtering

Content-Based Filtering leverages **BERT embeddings** to represent news articles and **FAISS** for approximate nearest neighbor search.

- **Workflow**:
  1. Generate embeddings for news articles using a pretrained BERT model.
  2. Index embeddings with FAISS for efficient similarity search.
  3. Retrieve similar articles based on a user’s reading history using cosine similarity.


 ---
 
### Clustering

To validate the embeddings generated for content-based filtering, **K-means clustering (k=3)** was performed on the news article embeddings. The goal was to ensure that similar articles were grouped together.

<br>

<p align="center">
  <img src="https://github.com/pippotek/Data-Mining-Project/blob/4ae958b80cb9b34f57bc81ef86b7611e491a8388/outputs/clusters_visualization_3.png?raw=true" width="512"/>  
</p>

---
## Results 
More about the results can be found in our report. An example of the **PowerBI dashboard** is showed below:  
<p align="center">
  <img src="https://github.com/pippotek/Dual-Recommendation-System/blob/30fe225d56cc30d55d1fe80eda03d217e6f89be0/outputs/gif%20data%20mining.mov?raw=true" width="512"/>  
</p>

---
## Authors
- [Sonia Borsi](https://github.com/SoniaBorsi)
- [Filippo Costamagna](https://github.com/pippotek)
- [Joaquin Lopez Calvo](https://github.com/JoaquinLCalvo)

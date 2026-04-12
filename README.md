A local, GPU-accelerated pipeline for predicting psychological item-level correlations. This project leverages Qwen-3 8B embeddings and XGBoost to recover the relational structure of inter-item correlations in psychological scales without manual survey administration.

📊 Data Sources
This project is built on the principles of Open Science, utilizing: 
- OpenPsychometrics Project: Large-scale raw survey data.
- R Package Ecosystem: Built-in datasets from psychometric packages (e.g., psych, psychTools).
- Multi-Study Integration: Aggregated open-access data from various clinical and personality research repositories.

🔄 Project Workflow
The pipeline follows a modular path from raw text to relational predictions:
1. Data Integration (item_integration/data_integration.R): Merging disparate open-access datasets into a unified interitem correlations dataframe and a list of the items with their descriptive statistics.
2. Semantic Analysis of item pairs (feature_engineering/crossencoder.ipynb): Generating Cross-Encoder scores (contradiction, logical friction) and sentiment comparisons using sentence-transformer models locally.
3. Sentiment Analysis of items (feature_engineering/semantic_analysis_items.ipynb): Runs transformer models locally to get scores of Sentiment/Emotion for each item.
4. Embedding Generation (feature_engineering/embed_items.R): Get embeddings for each distinct item in the dataset. Can run a local model (e.g. Qwen3-embedding:8b with Instruction) as well as make requests to either OpenAI or Gemini. 
5. Dimensionality Reduction (feature_engineering/autoencoder.ipynb): Autoencoding embeddings into x [default = 512] thematic clusters for efficient feature representation.
6. Correlation Modelling (modelling/cor_modelling.ipynb): Final predictive stage using optimal hyperparameters to generate out-of-sample predictions.

💻 Hardware & Performance
Designed for local, GPU-enabled processing to ensure data privacy and iteration speed:
- Primary Device: Optimized for NVIDIA CUDA-enabled GPUs (e.g., RTX 2080 Ti).
- Efficiency: Uses float32 precision and the XGBoost hist method to minimize VRAM footprint while maximizing training speed.
- HPT: Standalone Optuna module for multi-objective optimization (maximizing Pearson $r$ and minimizing RMSE).

📈 Current Performance
10-fold cross validation: r = 0.85; RMSE = .099

Further todos/Plans:
- Create a model predicting item-level descriptive statistics (e.g. IRT difficulty parameters)
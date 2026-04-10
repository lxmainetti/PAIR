# --- PSYCHOMETRIC ITEM EMBEDDING & FEATURE GENERATION ---
# Logic: Fetches LLM embeddings using either local models or proprietary models through their API.
# Workflow: Handles raw item text -> Embeddings

library(ollamar)
library(tidyverse)
library(furrr)
library(mall)
library(httr2)
library(arrow)
source("r_functions/get_embeddings.R")

# ---- Read in Data ----
# Read in the data with cross encoding and sentiment features
item_cors <- read_parquet(paste0("../../data/processed/item_correlations_with_cross.parquet"))
item_list <- read_parquet(paste0("../../data/processed/item_list_sentiment.parquet")) %>% 
  distinct(item, .keep_all = TRUE)
# ---- Use Functions to generate the Embeddings ----

# Get embeddings using local models (qwen3-embedding:8b usually performs best)
embeddings <- get_embeddings()

# API Call to Open AI or Google AI Studios for proprietary embeddings models
# embeddings <- get_embeddings_API(model = "text-embedding-3-large", dims = 3072)

# ---- Save Data ----
item_embeddings_df <- as.data.frame(do.call(rbind, embeddings)) %>% 
  bind_cols(item_list %>% select(item)) %>% 
  select(item, everything()) %>% 
  rename_with(~ str_replace(.x, "V", "emb"))

# Export item_list with each items embeddings to disk
write_parquet(item_embeddings_df, paste0("../../data/raw/", model_safe, "_embeddings_raw.parquet"))

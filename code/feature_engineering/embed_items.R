# --- PSYCHOMETRIC ITEM EMBEDDING ---
# Logic: Generate semantic embeddings for every unique item text via either a
#        local Ollama model or a hosted API (OpenAI / Google).
# Workflow: item_list parquet -> embed -> item-keyed embedding parquet.
#
# Runs once for each data split (training / holdout / validation). Switch the
# split by editing `split` below.

library(ollamar)
library(tidyverse)
library(furrr)
library(mall)
library(httr2)
library(arrow)
source("r_functions/get_embeddings.R")

# Which split to embed: "" (training), "holdout_" or "validation_"
split <- ""

# ---- Read in Data ----

item_cors <- read_parquet(paste0("../../data/processed/", split, "item_correlations_with_cross.parquet"))
item_list <- read_parquet(paste0("../../data/processed/", split, "item_list_sentiment.parquet")) %>%
  distinct(item, .keep_all = TRUE)

# ---- Generate Embeddings ----

# Local Ollama model (qwen3-embedding:8b + task-specific instruction works best
# for inter-item correlation prediction in my benchmarks)
embeddings <- get_embeddings()

# API alternative: OpenAI text-embedding-3-large or Gemini text-embedding-005
# embeddings <- get_embeddings_API(model = "text-embedding-3-large", dims = 3072)

# ---- Save Data ----

# Stack embedding vectors into a wide matrix, attach the item text, prefix
# columns with "emb" for downstream feature selectors.
item_embeddings_df <- as.data.frame(do.call(rbind, embeddings)) %>%
  bind_cols(item_list %>% select(item)) %>%
  select(item, everything()) %>%
  rename_with(~ str_replace(.x, "V", "emb"))

dir.create(paste0("../../data/raw/", model_safe), showWarnings = FALSE)
write_parquet(item_embeddings_df,
              paste0("../../data/raw/", model_safe, "/", split, "embeddings_raw.parquet"))

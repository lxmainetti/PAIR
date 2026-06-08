# --- EMBEDDING BACKENDS ---
# Logic: Two helper functions that turn item_list$item into a list of numeric
#        embedding vectors, either through a local Ollama model or through the
#        OpenAI / Google embedding API.
# Workflow: Called by embed_items.R; both write the raw embedding list to
#           data/raw/<model_safe>/embeddings_raw.rds as a side-effect.

library(dplyr)
library(ollamar)
library(furrr)
library(httr2)


# ---- Local Ollama embeddings ----

# Generate embeddings using a locally-served Ollama model. qwen3-embedding:8b
# with the task-specific instruction prefix is the best performer on the
# inter-item correlation task in my benchmarks.
get_embeddings <- function(model = "qwen3-embedding:8b",
                           include_instruction = TRUE) {

  if (!exists("item_list")) stop("Required object 'item_list' not found.")

  # Filename-safe model identifier shared with the caller via global assign
  model_safe <<- gsub(":", "-", model)
  model_safe <<- gsub("/", "", model_safe)

  # Override the path label when running qwen3 with the instruction prefix so
  # instructed and non-instructed runs don't overwrite each other on disk
  instruct_model <- str_detect(model, "qwen3") & include_instruction
  if(instruct_model) {
    model_safe <<- "qwen3-8b-with-instruction"
  }

  dir.create(paste0("../../data/clustered_embeddings/", model_safe), showWarnings = FALSE)
  embeddings_path <- paste0("../../data/raw/", model_safe, "/embeddings_raw.rds")

  # Build the model input: prepend Qwen3's recommended instruction prefix if
  # we're using the instructed variant; otherwise feed the raw item text
  if(instruct_model){
    task_desc <- "Given a psychometric scale item, represent its latent psychological constructs for predicting response correlations and factor structures."
    item_list <- item_list %>%
      mutate(input = paste0("Instruct: ", task_desc, "\nQuery: ", item))
  } else {
    item_list <- item_list %>% mutate(input = item)
  }

  # Embed in parallel via Ollama, save raw response to disk for re-use
  plan(multisession, workers = 16)
  embeddings_list <- future_map(
    item_list$input,
    function(x) embed(model = model, input = x),
    .progress = TRUE
  )

  embeddings_list <- map(embeddings_list, as.numeric)
  write_rds(embeddings_list, embeddings_path)

  embeddings_list
}


# ---- Hosted-API embeddings (OpenAI / Google) ----

# Generate embeddings using OpenAI or Gemini. Provider is detected from the
# model name. Output dimensionality is fixed for older OpenAI models, tunable
# from text-embedding-3-* onwards, and configurable on Gemini.
get_embeddings_API <- function(model = "text-embedding-3-large",
                               dims = 3072,
                               task_type = "SEMANTIC_SIMILARITY") {

  if (!exists("item_list")) stop("Required object 'item_list' not found.")
  items <- item_list %>% dplyr::pull(item)

  provider <- if (str_detect(model, "gemini")) "google" else "openai"
  model_safe <<- gsub("[:/]", "-", model)
  output_dir <- "../../data/raw/"
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  message(paste0("Starting ", provider, " embedding for ", length(items), " items..."))
  embeddings_list <- list()

  # ---- OpenAI ----
  if (provider == "openai") {
    api_key <- Sys.getenv("OPEN_AI_API_KEY")

    body_list <- list(model = model)
    # Only embedding-3 generation supports tunable output dims
    if (str_detect(model, "-3-")) body_list$dimensions <- dims

    # OpenAI accepts up to 2048 inputs per request; batch at 100 for safety
    item_batches <- split(items, ceiling(seq_along(items) / 100))

    for (batch in seq_along(item_batches)) {
      body <- body_list
      body$input <- item_batches[[batch]]

      resp <- request("https://api.openai.com/v1/embeddings") %>%
        req_headers(Authorization = paste("Bearer", api_key)) %>%
        req_body_json(body) %>%
        req_retry(max_tries = 3) %>%
        req_perform()

      res_data <- resp %>% resp_body_json()
      batch_results <- map(res_data$data, ~ as.numeric(.x$embedding))
      embeddings_list <- c(embeddings_list, batch_results)

      Sys.sleep(.3)  # rate-limit buffer
    }
  }

  # ---- Google (Gemini) ----
  if (provider == "google") {
    api_key <- Sys.getenv("GOOGLE_API_KEY")

    # Gemini batchEmbedContents tops out at ~50 inputs per request
    batches <- split(items, ceiling(seq_along(items) / 50))

    for (i in seq_along(batches)) {
      message(paste("Batch", i, "of", length(batches)))

      body <- list(requests = map(batches[[i]], ~ list(
        model = paste0("models/", model),
        content = list(parts = list(list(text = .x))),
        taskType = task_type,
        outputDimensionality = dims
      )))

      resp <- request(paste0("https://generativelanguage.googleapis.com/v1beta/models/",
                             model, ":batchEmbedContents")) %>%
        req_url_query(key = api_key) %>%
        req_body_json(body) %>%
        req_retry(max_tries = 3, backoff = ~ 5) %>%
        req_perform()

      res_data <- resp %>% resp_body_json()
      batch_vectors <- map(res_data$embeddings, ~ as.numeric(.x$values))
      embeddings_list <- c(embeddings_list, batch_vectors)

      Sys.sleep(0.3)
    }
  }

  # Key by item text and persist
  names(embeddings_list) <- items
  write_rds(embeddings_list, paste0(output_dir, "/", model_safe, "/embeddings_raw.rds"))
  message("SUCCESS: Embeddings saved to ", output_dir)

  return(embeddings_list)
}

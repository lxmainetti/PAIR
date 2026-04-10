library(dplyr)
library(ollamar)
library(furrr)
library(httr2)


# ---- Function Definitions ----

# Gets embeddings using local models (qwen3-embedding:8b usually performs best)
get_embeddings <- function(model = "qwen3-embedding:8b", 
                           include_instruction = TRUE) {
  
  # Ensure input source exists in environment
  if (!exists("item_list")) stop("Required object 'item_list' not found.")
  
  # Define directory and model identifiers
  model_safe <<- gsub(":", "-", model)
  model_safe <<- gsub("/", "", model_safe)
  
  # Modify Path if model is used with instruction
  instruct_model <- str_detect(model, "qwen3") & include_instruction
  if(instruct_model) {
    model_safe <<- "qwen3-8b-with-instruction"
  }
  
  # Create the output folder
  dir.create(paste0("../../data/clustered_embeddings/", model_safe), showWarnings = FALSE)
  
  # Define embeddings path
  embeddings_path <- paste0("../../data/raw/", model_safe, "_embeddings_raw.rds")
  
  # Create Instruction for qwen3 models if selected; else just give items as input
  if(instruct_model){
    task_desc <- "Given a psychometric scale item, represent its latent psychological construct for predicting response correlations and factor structures."
    item_list <- item_list %>% 
      mutate(input = paste0("Instruct: ", task_desc, "\nQuery: ", item))
  } else {
    item_list <- item_list %>% 
      mutate(input = item)
  }
  
  # If the embeddings do not already exist generate them; Else skip and read from disk
  if(!file.exists(embeddings_path)) {
    # Initialize parallel workers (16 threads)
    plan(multisession, workers = 16)
    
    # Execute parallel embedding generation via future_map
    embeddings_list <- future_map(
      item_list$input,
      function(x) {
        # Interface with local Ollama server
        res <- embed(model = model, input = x)
        return(res) # Pulls the numeric vector of embeddings
      },
      .progress = TRUE # Show a progress bar
    )
    # Cast response to numeric vectors and save to disk
    embeddings_list <- map(embeddings_list, as.numeric)
    write_rds(embeddings_list, embeddings_path)
    
  } else { 
    # Read embeddings if already generated once
    embeddings_list <- read_rds(embeddings_path)
  }
  # Return embeddings_list
  embeddings_list
}

# Gets embeddings using the API of Google or OpenAI
get_embeddings_API <- function(model = "text-embedding-3-large", 
                               dims = 3072, 
                               task_type = "SEMANTIC_SIMILARITY") {
  
  # Ensure input source exists in environment
  if (!exists("item_list")) stop("Required object 'item_list' not found.")
  items <- item_list %>% dplyr::pull(item)
  
  # Provider detection and path standardization
  provider <- if (str_detect(model, "gemini")) "google" else "openai"
  model_safe <<- gsub("[:/]", "-", model)
  output_dir <- "../../data/raw/"
  
  # Create the output folder for the embeddings if it doesnt exist yet
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  message(paste0("Starting ", provider, " embedding for ", length(items), " items..."))
  
  # Create empty list to bind embeddings into
  embeddings_list <- list()
  
  # --- PROVIDER: OPENAI ---
  if (provider == "openai") {
    api_key <- Sys.getenv("OPEN_AI_API_KEY")
    
    # Define the request, the model and its input
    body_list <- list(model = model)
    
    # If model version is 3 also specify how many dimensions (earlier versions have fixed dimensions)
    if (str_detect(model, "-3-")) {
      body_list$dimensions <- dims
    }
    
    # Split items into batches
    item_batches <- split(items, ceiling(seq_along(items) / 100))
    
    # Execute a single request for the entire input vector 
    # (openAI allows for up to 2048 inputs)
    for (batch in seq_along(item_batches)) {
      
      # Inject the current batch into the body list
      body <- body_list
      body$input <- item_batches[[batch]]
      
      # Send request to OpenAI API
      resp <- request("https://api.openai.com/v1/embeddings") %>%
        req_headers(Authorization = paste("Bearer", api_key)) %>%
        req_body_json(body) %>%
        req_retry(max_tries = 3) %>%
        req_perform()
      
      # extract numeric vectors from the response array
      res_data <- resp %>% resp_body_json()
      batch_results <- map(res_data$data, ~ as.numeric(.x$embedding))
      
      # Attach embeddings of the batch to the whole list
      embeddings_list <- c(embeddings_list, batch_results)
      
      Sys.sleep(.3) # Rate limiting
    }
  }
  
  # --- PROVIDER: GOOGLE (GEMINI) ---
  if (provider == "google") {
    api_key <- Sys.getenv("GOOGLE_API_KEY")
    
    # Gemini requires batching for multi-item requests
    batches <- split(items, ceiling(seq_along(items) / 50))
    
    # Iterative batching loop to handle Gemini's batchEmbedContents endpoint
    for (i in seq_along(batches)) {
      message(paste("Batch", i, "of", length(batches)))
      
      # Build request object matching Google's content-part structure
      body <- list(requests = map(batches[[i]], ~ list(
        model = paste0("models/", model),
        content = list(parts = list(list(text = .x))),
        taskType = task_type,
        outputDimensionality = dims
      )))
      
      # Execute batch request using v1beta API endpoint
      resp <- request(paste0("https://generativelanguage.googleapis.com/v1beta/models/", 
                             model, ":batchEmbedContents")) %>%
        req_url_query(key = api_key) %>%
        req_body_json(body) %>%
        req_retry(max_tries = 3, backoff = ~ 5) %>%
        req_perform()
      
      # Parse 'embeddings' list from response and append to the master list
      res_data <- resp %>% resp_body_json()
      batch_vectors <- map(res_data$embeddings, ~ as.numeric(.x$values))
      embeddings_list <- c(embeddings_list, batch_vectors)
      
      Sys.sleep(0.3) # API rate-limiting buffer
    }
  }
  
  # Key list by item and save to disk
  names(embeddings_list) <- items
  write_rds(embeddings_list, paste0(output_dir, model_safe, "_embeddings_raw.rds"))
  message("SUCCESS: Embeddings saved to ", output_dir)
  
  return(embeddings_list)
}

# ---- Psychometric Data Integration & Normalization Pipeline ----
# This script parses diverse scale data and codebooks to generate a unified 
# item-level correlation and descriptive statistics database.

library(tidyverse)
library(psychTools)
library(correlation)
library(stringr)
library(readxl)

dir.create("../../data/raw")

# ---- Global Data Accumulators ----
dat_cors <- tibble()
item_list <- tibble()

# ---- Core Processing Functions ----

# Appends processed item correlations to the global accumulator
bind_cors <- function(cor_items){
  dat_cors <<- dat_cors %>% bind_rows(cor_items)
}

# Calculates descriptive statistics and appends wording to the global item list
append_item_list <- function(data, item_wordings) {
  # Generate summary statistics (Mean, SD, Min, Max) for each item
  stats <- data %>%
    summarise(across(everything(), list(
      mean = ~ mean(.x, na.rm = TRUE),
      sd   = ~ sd(.x, na.rm = TRUE),
      max_scale = ~ max(.x, na.rm = TRUE),
      min_scale = ~ min(.x, na.rm = TRUE)
    ))) %>% 
    pivot_longer(
      everything(), 
      names_to = c("item_name", ".value"), 
      names_pattern = "(.*)_(mean|sd|max|min)"
    )
  
  # Merge scale metadata with statistical descriptors
  new_rows <- bind_cols(item_wordings, stats %>% select(-item_name))
  item_list <<- bind_rows(item_list, new_rows)
}

# Principal orchestration function for scale processing
intercorrelations <- function(data, dic_col, testing = FALSE){
  dat <- data
  colnames(dat) <- dic_col
  
  if(!testing)
    append_item_list(data, dic_col %>% as_tibble() %>% rename(item = value))
  
  # Compute pairwise Pearson correlations, excluding self-correlations
  cor_items <- cor(dat, use = "pairwise.complete.obs") %>%
    as_tibble(rownames = "Parameter1") %>%
    pivot_longer(-Parameter1, names_to = "Parameter2", values_to = "r") %>%
    filter(Parameter1 < Parameter2)
  
  if(!testing)
    bind_cors(cor_items)
  else
    cor_items
}

# ---- Codebook Extraction Utilities ----

# Robust extractor for Q-prefixed items (standard psychometric format)
extract_items_robust <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  
  # Target lines specifically starting with 'Q' followed by item identifiers
  item_lines <- lines[str_detect(lines, "^Q[0-9]+[\\.\\s\\t]+")]
  item_ids <- str_extract(item_lines, "Q[0-9]+")
  
  # Sanitize item text: strip IDs, fix encoding artifacts
  clean_items <- str_remove(item_lines, "^Q[0-9]+[\\.\\s\\t]+") %>% str_trim()
  clean_items <- str_replace_all(clean_items, "([a-z])\\?([a-z])", "\\1'\\2")
  clean_items <- str_replace_all(clean_items, "[^ -~]", "'")
  clean_items <- str_replace_all(clean_items, "\\s+", " ")
  
  names(clean_items) <- item_ids
  return(clean_items)
}

# Specialized extractor for HEXACO scale nomenclature
extract_hexaco_items <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  lines <- str_replace_all(lines, "[^ -~]", "'")
  
  # Regex targeting HEXACO trait-prefix format (e.g., OCrea10)
  pattern <- "^([A-Z][A-Za-z]+[0-9]+)\\s+(.+)$"
  item_lines <- lines[str_detect(lines, pattern)]
  matches <- str_match(item_lines, pattern)
  
  item_ids <- matches[,2]
  item_text <- str_trim(matches[,3])
  
  # Filter metadata and validation columns
  is_hexaco_item <- !str_detect(item_ids, "^V[0-9]") & 
    !str_detect(item_text, "ISO country code|seconds from")
  
  final_items <- item_text[is_hexaco_item]
  names(final_items) <- item_ids[is_hexaco_item]
  return(final_items)
}

# Specialized extractor for RIASEC occupational interest items
extract_riasec_items <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  item_lines <- lines[str_detect(lines, "^[RIASEC][1-8]\\t")]
  parts <- str_split_fixed(item_lines, "\\t", 2)
  
  item_ids <- str_trim(parts[,1])
  item_text <- str_trim(parts[,2]) %>% str_remove("^[\\.\\s]+")
  
  names(item_text) <- item_ids
  return(item_text)
}

# Utility for subsampling large datasets to maintain computational efficiency
shorten <- function(data){
  data %>% slice_sample(n=2000)
}

# ---- Execution: Standardized Scales (psychTools) ----
# Eysenck Personality Inventory (EPI)
intercorrelations(psychTools::epi, epi.dictionary$Content)

# Big Five Inventory (BFI)
intercorrelations(bfi, bfi.dictionary$Item)

# Motivational State Questionnaire (MSQ) - Cleaning item labels
msq %>% select(-c(EA:exper)) %>% 
  rename_with(.cols = contains("."), ~ stringr::str_replace_all(.x, "[.]", "-")) %>% 
  intercorrelations(., colnames(.))

# SAPA Project Personality Inventory (SPI) - Truncating dictionary metadata
spi %>% 
  select(-c(age:ER)) %>% 
  intercorrelations(., spi.dictionary %>% slice(11:nrow(.)) %>% pull(item))

# Athenstaedt Gender Role Self-Concept
intercorrelations(Athenstaedt %>% as_tibble() %>% select(starts_with("V")),
                  psychTools::Athenstaedt.dictionary$Item[2:75])

# ---- EXECUTION: CUSTOM EXTERNAL DATASETS ----
# Define standardized Paths
cb <- "/codebook.txt"
path_to_scales <- "../../data/scales/"
data_path <- "/data.csv"

# Humor Styles Questionnaire (HSQ)
humor <- read_csv(paste0(path_to_scales, "humor", data_path)) %>% select(Q1:Q32)
humor_items <- extract_items_robust(paste0(path_to_scales, "humor", cb))
intercorrelations(humor, humor_items)

# Taylor Manifest Anxiety Scale (TMA)
TMA <- read_csv(paste0(path_to_scales, "TMA", data_path)) %>% select(starts_with("Q"))
TMA_items <- extract_items_robust(paste0(path_to_scales, "TMA", cb))
intercorrelations(TMA, TMA_items)

# HEXACO-60 Personality Inventory
hexaco <- read_tsv(paste0(path_to_scales, "HEXACO", data_path)) %>% select(-c("V1", "V2", "country", "elapse"))
hexaco_items <- extract_hexaco_items(paste0(path_to_scales, "hexaco", cb))
item_cols <- colnames(hexaco)[!colnames(hexaco) %in% c("age", "gender", "accuracy", "country", "elapse", "V1", "V2")]
intercorrelations(hexaco %>% shorten(), hexaco_items[item_cols])

# RIASEC Holland Occupational Themes
riasec <- read_tsv(paste0(path_to_scales, "riasec", data_path)) %>% select(R1:C8)
riasec_items <- extract_riasec_items(paste0(path_to_scales, "riasec", cb))
intercorrelations(riasec %>% shorten(), riasec_items)

# Consideration of Future Consequences Scale (CFCS)
CFCS_items <- extract_items_robust(paste0(path_to_scales, "CFCS", cb))
CFCS <- read_tsv(paste0(path_to_scales, "CFCS", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(CFCS, CFCS_items)

# Depression Anxiety Stress Scales (DASS)
DASS_items <- extract_items_robust(paste0(path_to_scales, "DASS", cb))
DASS <- read_tsv(paste0(path_to_scales, "DASS", data_path)) %>% select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(DASS, DASS_items)

# Experiences in Close Relationships (ECR)
ECR_items <- extract_items_robust(paste0(path_to_scales, "ECR", cb))
ECR <- read_csv(paste0(path_to_scales, "ECR", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(ECR, ECR_items)

# Empathizing/Systemizing Quotient (EQSQ)
EQSQ_items <- extract_items_robust(paste0(path_to_scales, "EQSQ", cb))
EQSQ <- read_tsv(paste0(path_to_scales, "EQSQ", data_path)) %>% select(starts_with("E"), starts_with("S"), -SQ, -EQ) %>% shorten()
intercorrelations(EQSQ, EQSQ_items)

# Generic Conspiracist Beliefs Scale (GCBS)
GCBS_items <- extract_items_robust(paste0(path_to_scales, "GCBS", cb))
GCBS <- read_csv(paste0(path_to_scales, "GCBS", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(GCBS, GCBS_items)

# Kentucky Inventory of Mindfulness Skills (KIMS)
KIMS_items <- extract_items_robust(paste0(path_to_scales, "KIMS", cb))
KIMS <- read_csv(paste0(path_to_scales, "KIMS", data_path)) %>% select(starts_with("Q"))
intercorrelations(KIMS, KIMS_items)

# Machiavellianism (MACH-IV)
MACH_items <- extract_items_robust(paste0(path_to_scales, "MACH", cb))
MACH <- read_tsv(paste0(path_to_scales, "MACH", data_path)) %>% select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(MACH, MACH_items)

# Multidimensional General Knowledge Test (MGKT)
MGKT_items <- extract_items_robust(paste0(path_to_scales, "MGKT", cb))
MGKT <- read_csv(paste0(path_to_scales, "MGKT", data_path)) %>% 
  select(starts_with("Q") & ends_with("A")) %>% 
  shorten() %>% 
  mutate(across(everything(), ~ nchar(.x) / 2)) # Numeric normalization
intercorrelations(MGKT, MGKT_items)

# Narcissistic Personality Adjective Checklist (NPAS)
NPAS_items <- extract_items_robust(paste0(path_to_scales, "NPAS", cb))
NPAS <- read_tsv(paste0(path_to_scales, "NPAS", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(NPAS, NPAS_items)

# Rosenberg Self-Esteem Scale (RSE)
RSE_items <- extract_items_robust(paste0(path_to_scales, "RSE", cb))
RSE <- read_tsv(paste0(path_to_scales, "RSE", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(RSE, RSE_items)

# Right-Wing Authoritarianism Scale (RWAS)
RWAS_items <- extract_items_robust(paste0(path_to_scales, "RWAS", cb))
RWAS <- read_csv(paste0(path_to_scales, "RWAS", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(RWAS, RWAS_items)

# Self-Compassion Scale (SCS)
SCS_items <- extract_items_robust(paste0(path_to_scales, "SCS", cb))
SCS <- read_csv(paste0(path_to_scales, "SCS", data_path)) %>% select(starts_with("Q")) %>% shorten()
intercorrelations(SCS, SCS_items)

# Self-Efficacy Fragility (Excel Import; own study, material on OSF)
path_SE <- paste0(path_to_scales, "self-efficacy-security/data_final.xlsx")
SEFrag <- readxl::read_xlsx(path_SE) %>% select(starts_with("item_")) %>% rename_with(~ str_remove(.x, "item_"))
intercorrelations(SEFrag, colnames(SEFrag))

# Adult Multidimensional Independence (AMBI)
AMBI_items <- extract_items_robust(paste0(path_to_scales, "AMBI", cb))
AMBI <- read_tsv(paste0(path_to_scales, "AMBI", data_path)) %>% select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(AMBI, AMBI_items)

# 16 Personality Factor Questionnaire (16PF)
PF16_items <- extract_items_robust(paste0(path_to_scales, "16PF", cb))
PF16 <- read_tsv(paste0(path_to_scales, "16PF", data_path)) %>% 
  select(A1:P9) %>% 
  shorten() %>% 
  mutate(across(everything(), ~ na_if(.x, 0))) # Masking zero values as NA
intercorrelations(PF16, PF16_items)

# Personality Inventory for DSM-5 https://osf.io/6hwzk/overview
PID_items <- readxl::read_xlsx(paste0(path_to_scales, "PID/codebook.xlsx")) %>% 
  filter(str_detect(variable, "pid")) %>% 
  dplyr::pull(label)
PID <- read_csv(paste0(path_to_scales, "PID", data_path)) %>% 
  select(starts_with("pid"))
intercorrelations(PID, PID_items)

# Symptom Checklist 90 Revised https://osf.io/q5rgb/overview
SCL90R_items <- extract_items_robust(paste0(path_to_scales, "SCL90R", cb))
SCL90R <- read_xlsx(paste0(path_to_scales, "SCL90R/data.xlsx")) %>% 
  mutate(across(everything(), as.numeric))
intercorrelations(SCL90R, SCL90R_items)

# Psychological Strain Scales https://osf.io/q5rgb/overview
PSS_items <- extract_items_robust(paste0(path_to_scales, "PSS", cb))
PSS <- read_csv(paste0(path_to_scales, "PSS/data.csv")) %>% 
  mutate(across(everything(), as.numeric))
intercorrelations(SCL90R, SCL90R_items)

# Comprehensive Autistic Inventory and Adult ADHD Self-Report Scale https://osf.io/qtngb/overview
CATI_ASRS_items <- extract_items_robust(paste0(path_to_scales, "CATI_ASRS", cb))
CATI_ASRS <- read_csv(paste0(path_to_scales, "CATI_ASRS/data.csv"), skip = 1) %>% 
  select(contains("ASRS"), contains("CATI"), -`Education level`)
intercorrelations(CATI_ASRS, CATI_ASRS_items)

# 
# # Dataverse Data from Goldberg (2018; https://doi.org/10.7910/DVN/ZNGS1K)
# goldberg_data <- read_table(paste0(path_to_scales, "360_words/data.tab")) %>% select(-ID)
# goldberg_items <- extract_items_robust(paste0(path_to_scales, "360_words/codebook.txt")) %>% 
#   str_to_lower() %>% str_trim()
# 
# goldberg_items_std <- tibble(item = goldberg_items) %>% 
#   mutate(item = paste0("I am ", item)) %>% 
#   dplyr::pull(item)
# 
# intercorrelations(goldberg_data, goldberg_items_std)
# 
# # Dataverse Data from Saucier (2018; https://doi.org/10.7910/DVN/ZNGS1K)
# saucier_data <- haven::read_sav(paste0(path_to_scales, "525_words/data.sav")) %>% 
#   haven::zap_labels() %>% 
#   select(-id)
# saucier_items <- extract_items_robust(paste0(path_to_scales, "525_words/codebook.txt")) %>% 
#   str_to_lower() %>% 
#   str_trim()
# 
# saucier_items_std <- tibble(item = saucier_items) %>% 
#   mutate(item = paste0("I am ", item)) %>% 
#   dplyr::pull(item)
# 
# intercorrelations(saucier_data, saucier_items_std)


# ---- FINAL EXPORT ----
arrow::write_parquet(dat_cors %>% 
                       distinct(Parameter1, Parameter2, .keep_all = TRUE), 
                     "../../data/raw/item_correlations.parquet")
arrow::write_parquet(item_list %>% distinct(item, .keep_all = TRUE),
                     "../../data/raw/item_list.parquet")

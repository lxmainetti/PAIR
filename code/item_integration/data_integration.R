# --- PSYCHOMETRIC DATA INTEGRATION & NORMALIZATION ---
# Logic: Parses heterogeneous open-access scale data + codebooks into a unified
#        item-level correlation table and an item list with descriptive stats.
# Workflow: Raw CSV/TSV/SAV/RDS per scale -> per-scale intercorrelations ->
#           append to global accumulators -> export train + holdout + validation
#           parquet files.

library(tidyverse)
library(psychTools)
library(correlation)
library(stringr)
library(readxl)
library(haven)

# ---- Setup ----

dir.create("../../data/raw", showWarnings = FALSE)
options(vroom.tempdir = tempdir())
options(vroom.altrep = FALSE)

# Global accumulators (filled by intercorrelations() / append_item_list())
dat_cors <- tibble()
item_list <- tibble()

# Standardized paths used by the per-scale blocks
cb <- "/codebook.txt"
path_to_scales <- "../../data/scales/"
data_path <- "/data.csv"

# ---- Core Processing Functions ----

# Appends pairwise correlations of one scale to the global accumulator
bind_cors <- function(cor_items){
  dat_cors <<- dat_cors %>% bind_rows(cor_items)
}

# Computes per-item descriptives (mean/sd on a rescaled 0-10 scale) and binds
# them onto the global item list together with the item wording
append_item_list <- function(data, item_wordings) {
  stats <- data %>%
    mutate(across(everything(), ~ {
      orig_min <- min(.x, na.rm = TRUE)
      orig_max <- max(.x, na.rm = TRUE)
      rescale_to_10(.x, orig_min, orig_max)
    })) %>%
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

  new_rows <- bind_cols(item_wordings, stats %>% select(-item_name))
  item_list <<- bind_rows(item_list, new_rows)
}

# Main per-scale entry point: relabels columns with their item text, computes
# pairwise Pearson r (long format, upper-triangle only) and pushes both
# correlations and item-list metadata into the global accumulators.
# testing = TRUE returns the correlation tibble without mutating globals.
intercorrelations <- function(data, dic_col, testing = FALSE){
  dat <- data
  colnames(dat) <- dic_col

  if(!testing)
    append_item_list(data, dic_col %>% as_tibble() %>% rename(item = value))

  cor_items <- cor(dat, use = "pairwise.complete.obs") %>%
    as_tibble(rownames = "Parameter1") %>%
    pivot_longer(-Parameter1, names_to = "Parameter2", values_to = "r") %>%
    filter(Parameter1 < Parameter2)

  if(!testing)
    bind_cors(cor_items)
  else
    cor_items
}

# Map an arbitrary numeric column to a 1-10 scale (used before computing M/SD
# so item-level descriptives are comparable across response formats).
rescale_to_10 <- function(x, old_min, old_max) {
  ((x - old_min) / (old_max - old_min)) * (10 - 1) + 1
}

# ---- Codebook Extraction Utilities ----

# Generic extractor for Q-prefixed item lines (the OpenPsychometrics format)
extract_items_robust <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")

  item_lines <- lines[str_detect(lines, "^Q[0-9]+[\\.\\s\\t]+")]
  item_ids <- str_extract(item_lines, "Q[0-9]+")

  # Strip IDs, repair encoding artifacts (latin1 quotes -> ASCII apostrophes)
  clean_items <- str_remove(item_lines, "^Q[0-9]+[\\.\\s\\t]+") %>% str_trim()
  clean_items <- str_replace_all(clean_items, "([a-z])\\?([a-z])", "\\1'\\2")
  clean_items <- str_replace_all(clean_items, "[^ -~]", "'")
  clean_items <- str_replace_all(clean_items, "\\s+", " ")

  names(clean_items) <- item_ids
  return(clean_items)
}

# HEXACO codebook uses trait-prefix IDs (e.g. OCrea10) rather than Q-numbers
extract_hexaco_items <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  lines <- str_replace_all(lines, "[^ -~]", "'")

  pattern <- "^([A-Z][A-Za-z]+[0-9]+)\\s+(.+)$"
  item_lines <- lines[str_detect(lines, pattern)]
  matches <- str_match(item_lines, pattern)

  item_ids <- matches[,2]
  item_text <- str_trim(matches[,3])

  # Filter out metadata / validation columns mis-matched by the pattern
  is_hexaco_item <- !str_detect(item_ids, "^V[0-9]") &
    !str_detect(item_text, "ISO country code|seconds from")

  final_items <- item_text[is_hexaco_item]
  names(final_items) <- item_ids[is_hexaco_item]
  return(final_items)
}

# RIASEC codebook is tab-separated with single-letter trait + index IDs (R1...C8)
extract_riasec_items <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "latin1")
  item_lines <- lines[str_detect(lines, "^[RIASEC][1-8]\\t")]
  parts <- str_split_fixed(item_lines, "\\t", 2)

  item_ids <- str_trim(parts[,1])
  item_text <- str_trim(parts[,2]) %>% str_remove("^[\\.\\s]+")

  names(item_text) <- item_ids
  return(item_text)
}

# Subsample large datasets to keep the correlation step tractable
shorten <- function(data){
  data %>% slice_sample(n=2000)
}

# Reset accumulators between train / holdout / validation passes
reset_accumulators <- function(){
  dat_cors <<- tibble()
  item_list <<- tibble()
}

# Write current accumulators to disk as parquet with the given prefix
export_accumulators <- function(prefix = ""){
  arrow::write_parquet(
    dat_cors %>% distinct(Parameter1, Parameter2, .keep_all = TRUE),
    paste0("../../data/raw/", prefix, "item_correlations.parquet")
  )
  arrow::write_parquet(
    item_list %>% distinct(item, .keep_all = TRUE),
    paste0("../../data/raw/", prefix, "item_list.parquet")
  )
}

# ===========================================================================
# TRAINING SET
# ===========================================================================

# ---- Scales bundled with psychTools ----

# Eysenck Personality Inventory
intercorrelations(psychTools::epi, epi.dictionary$Content)

# Big Five Inventory
intercorrelations(bfi, bfi.dictionary$Item)

# Motivational State Questionnaire (drop summary scores, fix dot-separated IDs)
msq %>% select(-c(EA:exper)) %>%
  rename_with(.cols = contains("."), ~ stringr::str_replace_all(.x, "[.]", "-")) %>%
  intercorrelations(., colnames(.))

# SAPA Project Personality Inventory (skip demographics + first 10 dict rows)
spi %>%
  select(-c(age:ER)) %>%
  intercorrelations(., spi.dictionary %>% slice(11:nrow(.)) %>% pull(item))

# Athenstaedt Gender Role Self-Concept
intercorrelations(Athenstaedt %>% as_tibble() %>% select(starts_with("V")),
                  psychTools::Athenstaedt.dictionary$Item[2:75])

# ---- OpenPsychometrics scales (Q-prefixed codebook format) ----

# Humor Styles Questionnaire
humor <- read_csv(paste0(path_to_scales, "humor", data_path)) %>% select(Q1:Q32)
humor_items <- extract_items_robust(paste0(path_to_scales, "humor", cb))
intercorrelations(humor, humor_items)

# Taylor Manifest Anxiety Scale
TMA <- read_csv(paste0(path_to_scales, "TMA", data_path)) %>% select(starts_with("Q"))
TMA_items <- extract_items_robust(paste0(path_to_scales, "TMA", cb))
intercorrelations(TMA, TMA_items)

# HEXACO-60
hexaco <- read_tsv(paste0(path_to_scales, "HEXACO", data_path)) %>%
  select(-c("V1", "V2", "country", "elapse"))
hexaco_items <- extract_hexaco_items(paste0(path_to_scales, "hexaco", cb))
item_cols <- colnames(hexaco)[!colnames(hexaco) %in%
                                c("age", "gender", "accuracy", "country", "elapse", "V1", "V2")]
intercorrelations(hexaco %>% shorten(), hexaco_items[item_cols])

# RIASEC Holland Occupational Themes
riasec <- read_tsv(paste0(path_to_scales, "riasec", data_path)) %>% select(R1:C8)
riasec_items <- extract_riasec_items(paste0(path_to_scales, "riasec", cb))
intercorrelations(riasec %>% shorten(), riasec_items)

# Consideration of Future Consequences Scale
CFCS_items <- extract_items_robust(paste0(path_to_scales, "CFCS", cb))
CFCS <- read_tsv(paste0(path_to_scales, "CFCS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(CFCS, CFCS_items)

# Depression Anxiety Stress Scales (only response columns end with "A")
DASS_items <- extract_items_robust(paste0(path_to_scales, "DASS", cb))
DASS <- read_tsv(paste0(path_to_scales, "DASS", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(DASS, DASS_items)

# Experiences in Close Relationships
ECR_items <- extract_items_robust(paste0(path_to_scales, "ECR", cb))
ECR <- read_csv(paste0(path_to_scales, "ECR", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(ECR, ECR_items)

# Empathizing / Systemizing Quotient (exclude scale totals SQ, EQ)
EQSQ_items <- extract_items_robust(paste0(path_to_scales, "EQSQ", cb))
EQSQ <- read_tsv(paste0(path_to_scales, "EQSQ", data_path)) %>%
  select(starts_with("E"), starts_with("S"), -SQ, -EQ) %>% shorten()
intercorrelations(EQSQ, EQSQ_items)

# Generic Conspiracist Beliefs Scale
GCBS_items <- extract_items_robust(paste0(path_to_scales, "GCBS", cb))
GCBS <- read_csv(paste0(path_to_scales, "GCBS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(GCBS, GCBS_items)

# Kentucky Inventory of Mindfulness Skills
KIMS_items <- extract_items_robust(paste0(path_to_scales, "KIMS", cb))
KIMS <- read_csv(paste0(path_to_scales, "KIMS", data_path)) %>% select(starts_with("Q"))
intercorrelations(KIMS, KIMS_items)

# Machiavellianism (MACH-IV)
MACH_items <- extract_items_robust(paste0(path_to_scales, "MACH", cb))
MACH <- read_tsv(paste0(path_to_scales, "MACH", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(MACH, MACH_items)

# Multidimensional General Knowledge Test (Q*A columns store raw text answers;
# nchar / 2 converts the comma-encoded score into a numeric value)
MGKT_items <- extract_items_robust(paste0(path_to_scales, "MGKT", cb))
MGKT <- read_csv(paste0(path_to_scales, "MGKT", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>%
  shorten() %>%
  mutate(across(everything(), ~ nchar(.x) / 2))
intercorrelations(MGKT, MGKT_items)

# Narcissistic Personality Adjective Checklist
NPAS_items <- extract_items_robust(paste0(path_to_scales, "NPAS", cb))
NPAS <- read_tsv(paste0(path_to_scales, "NPAS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(NPAS, NPAS_items)

# Rosenberg Self-Esteem
RSE_items <- extract_items_robust(paste0(path_to_scales, "RSE", cb))
RSE <- read_tsv(paste0(path_to_scales, "RSE", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(RSE, RSE_items)

# Right-Wing Authoritarianism
RWAS_items <- extract_items_robust(paste0(path_to_scales, "RWAS", cb))
RWAS <- read_csv(paste0(path_to_scales, "RWAS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(RWAS, RWAS_items)

# Self-Compassion Scale
SCS_items <- extract_items_robust(paste0(path_to_scales, "SCS", cb))
SCS <- read_csv(paste0(path_to_scales, "SCS", data_path)) %>%
  select(starts_with("Q")) %>% shorten()
intercorrelations(SCS, SCS_items)

# Adult Multidimensional Independence
AMBI_items <- extract_items_robust(paste0(path_to_scales, "AMBI", cb))
AMBI <- read_tsv(paste0(path_to_scales, "AMBI", data_path)) %>%
  select(starts_with("Q") & ends_with("A")) %>% shorten()
intercorrelations(AMBI, AMBI_items)

# 16 Personality Factor Questionnaire (0 codes missing -> set to NA)
PF16_items <- extract_items_robust(paste0(path_to_scales, "16PF", cb))
PF16 <- read_tsv(paste0(path_to_scales, "16PF", data_path)) %>%
  select(A1:P9) %>%
  shorten() %>%
  mutate(across(everything(), ~ na_if(.x, 0)))
intercorrelations(PF16, PF16_items)

# ---- Additional OSF / OpenPsychometrics scales (mixed file formats) ----

# Self-Efficacy Fragility (own study, materials on OSF)
SEFrag <- readxl::read_xlsx(paste0(path_to_scales, "self-efficacy-security/data_final.xlsx")) %>%
  select(starts_with("item_")) %>%
  rename_with(~ str_remove(.x, "item_"))
intercorrelations(SEFrag, colnames(SEFrag))

# Personality Inventory for DSM-5 (https://osf.io/6hwzk/overview)
PID_items <- readxl::read_xlsx(paste0(path_to_scales, "PID/codebook.xlsx")) %>%
  filter(str_detect(variable, "pid")) %>%
  dplyr::pull(label)
PID <- read_csv(paste0(path_to_scales, "PID", data_path)) %>% select(starts_with("pid"))
intercorrelations(PID, PID_items)

# Symptom Checklist 90 Revised (https://osf.io/q5rgb/overview)
SCL90R_items <- extract_items_robust(paste0(path_to_scales, "SCL90R", cb))
SCL90R <- read_xlsx(paste0(path_to_scales, "SCL90R/data.xlsx")) %>%
  mutate(across(everything(), as.numeric))
intercorrelations(SCL90R, SCL90R_items)

# Psychological Strain Scales (https://osf.io/q5rgb/overview)
PSS_items <- extract_items_robust(paste0(path_to_scales, "PSS", cb))
PSS <- read_csv(paste0(path_to_scales, "PSS/data.csv")) %>%
  mutate(across(everything(), as.numeric))
intercorrelations(PSS, PSS_items)

# Comprehensive Autistic Inventory + Adult ADHD Self-Report (https://osf.io/qtngb/overview)
CATI_ASRS_items <- extract_items_robust(paste0(path_to_scales, "CATI_ASRS", cb))
CATI_ASRS <- read_csv(paste0(path_to_scales, "CATI_ASRS/data.csv"), skip = 1) %>%
  select(contains("ASRS"), contains("CATI"), -`Education level`)
intercorrelations(CATI_ASRS, CATI_ASRS_items)

# General Attitudes towards Artificial Intelligence Scale (SPSS .sav with value/
# format labels; zap_* strips SPSS metadata that would break downstream numerics)
GAAIS <- haven::read_sav(paste0(path_to_scales, "GAAIS/data.sav")) %>%
  select(Pos1:Trust, -starts_with("BLANK")) %>%
  select(-c(21:57, 78:87))
colnames(GAAIS) <- sapply(GAAIS, function(x)
  attr(x, "label") %||% colnames(GAAIS)[which(sapply(GAAIS, identical, x))])
GAAIS <- GAAIS %>% zap_labels() %>% zap_label() %>% zap_formats() %>% zap_widths()
intercorrelations(GAAIS, colnames(GAAIS))

# Chronic Pain - Emotional / Trauma Survey
C_PETS_items <- extract_items_robust(paste0(path_to_scales, "C-PETS", cb))
C_PETS <- read_csv(paste0(path_to_scales, "C-PETS", data_path), skip = 2) %>% select(-c(1:4))
intercorrelations(C_PETS, C_PETS_items)

# Emotion Processes in Therapy-Engaged Patient Scale
EPTEPS_items <- extract_items_robust(paste0(path_to_scales, "EPTEPS", cb))
EPTEPS <- read_csv(paste0(path_to_scales, "EPTEPS", data_path)) %>% select(UE1:MO5)
intercorrelations(EPTEPS, EPTEPS_items)

# Vanity Scale (reverse-code items ending in "R" on a 1-5 Likert)
Vanity_Scale_items <- extract_items_robust(paste0(path_to_scales, "Vanity_Scale", cb))
Vanity_Scale <- read_sav(paste0(path_to_scales, "Vanity_Scale/data.sav")) %>%
  select(I1:I22) %>%
  zap_formats() %>%
  mutate(across(ends_with("R"), ~ 6 - .x))
intercorrelations(Vanity_Scale, Vanity_Scale_items)

# Multidimensional Self-Concept Questionnaire
MSSCQ_items <- extract_items_robust(paste0(path_to_scales, "MSSCQ", cb))
MSSCQ <- read_tsv(paste0(path_to_scales, "MSSCQ", data_path)) %>% select(starts_with("Q"))
intercorrelations(MSSCQ, MSSCQ_items)

# Hypersensitive Narcissism + Dirty Dozen
HSNS_DD_items <- extract_items_robust(paste0(path_to_scales, "HSNS+DD", cb))
HSNS_DD <- read_tsv(paste0(path_to_scales, "HSNS+DD", data_path)) %>%
  select(starts_with("H"), starts_with("D"))
intercorrelations(HSNS_DD, HSNS_DD_items)

# Short Dark Triad
SD3_items <- extract_items_robust(paste0(path_to_scales, "SD3", cb))
SD3 <- read_tsv(paste0(path_to_scales, "SD3", data_path)) %>% select(-country, -source)
intercorrelations(SD3, SD3_items)

# Bainbridge mega-study s1 (item texts via the authors' RDS label dict, with
# "I am someone who - X" / "I - X" prefixes flattened into proper sentences)
labels_bb <- read_rds(paste0(path_to_scales, "Bainbridge/label.rds"))
bainbridge <- read_csv(paste0(path_to_scales, "Bainbridge", data_path)) %>%
  select(-c(1, ac_1:Consent_T_Click.Count))
bainbridge_items <- unname(labels_bb$s1[colnames(bainbridge)]) %>%
  str_replace(" - ", " ") %>%
  str_to_sentence()
intercorrelations(bainbridge, bainbridge_items)

# SAPA 696-item public release
sapa_items <- read_csv(paste0(path_to_scales, "SAPA/iteminfo696.csv"),
                       locale = locale(encoding = "latin1")) %>%
  select(Item) %>% dplyr::pull()
sapa <- read_csv(paste0(path_to_scales, "SAPA/data.csv")) %>% select(-c(RID:p2occIncomeEst))
intercorrelations(sapa, sapa_items)

# ---- Export training set ----

export_accumulators(prefix = "")

# ===========================================================================
# HOLDOUT SET (Bainbridge s2 - cross-scale, item-disjoint from training)
# ===========================================================================

reset_accumulators()

bainbridge_holdout <- read_csv(paste0(path_to_scales, "bainbridge_2", data_path)) %>%
  select(-c(1, ac_1:DB_T_Click.Count))

stopifnot(all(colnames(bainbridge_holdout) %in% names(labels_bb$s2)))
bainbridge_holdout_items <- unname(labels_bb$s2[colnames(bainbridge_holdout)]) %>%
  str_replace(" - ", " ") %>%
  str_to_sentence()
intercorrelations(bainbridge_holdout, bainbridge_holdout_items)

export_accumulators(prefix = "holdout_")

# ===========================================================================
# VALIDATION SET (SurveyBot validation study - completely separate scales)
# ===========================================================================

reset_accumulators()

# Item wordings live in SPSS variable labels; pattern strips the "TraitX: " prefix
sb_val <- read_rds(paste0(path_to_scales, "surveybot_val_study/data.rds")) %>%
  as_tibble() %>%
  select(AAID_01:BFI10)

sb_val_items <- map_chr(sb_val, ~ attr(.x, "label") %||% NA_character_) %>%
  str_remove("^.*: ")

sb_val <- sb_val %>% zap_formats() %>% zap_label() %>% zap_labels()
intercorrelations(sb_val, sb_val_items)

export_accumulators(prefix = "validation_")

# --- CausalImpact pipeline (version-safe) ------------------------------------
# Run from REPO ROOT:
#   Rscript scripts/causal_impact.R

need <- c("MatchIt","CausalImpact","tidyverse","ggplot2","zoo","broom","bsts","jsonlite")
new  <- setdiff(need, rownames(installed.packages()))
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
  library(MatchIt)
  library(CausalImpact)
  library(tidyverse)
  library(ggplot2)
  library(zoo)
  library(broom)
  library(bsts)
  library(jsonlite)
})

# --- Config ------------------------------------------------------------------
data_path   <- "data/ontario_cases.csv"
policy_date <- as.Date("2021-02-01")

fig_dir <- "figures"; dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
res_dir <- "results"; dir.create(res_dir,  showWarnings = FALSE, recursive = TRUE)
plot_path   <- file.path(fig_dir, "fig_causalimpact.png")
summary_txt <- file.path(res_dir, "causalimpact_summary.txt")

# --- Load & clean ------------------------------------------------------------
stopifnot(file.exists(data_path))
df <- read.csv(data_path, stringsAsFactors = FALSE)

# normalize strings "NA" -> NA and make numerics numeric
df[] <- lapply(df, function(x) if (is.character(x)) dplyr::na_if(x, "NA") else x)
num_cols <- intersect(c("incidence","treated"), names(df))
for (nm in num_cols) df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
df <- tidyr::drop_na(df)

need_cols <- c("week","region","incidence","treated")
missing   <- setdiff(need_cols, names(df))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse=", "))

df <- df %>%
  mutate(
    week    = as.Date(week),
    region  = as.factor(region),
    treated = as.integer(treated),
    Post    = as.integer(week >= policy_date)
  )

if (any(is.na(df$week))) stop("NA in week after as.Date(); check date format.")
if (!all(df$treated %in% c(0,1))) stop("treated must be 0/1.")

# --- Matching on pre-policy baseline ----------------------------------------
pre_df <- df %>% filter(week < policy_date)
if (nrow(pre_df) == 0L) stop("No pre-policy rows before ", policy_date)

pre_baseline <- pre_df %>%
  group_by(region) %>%
  summarize(mean_incidence = mean(incidence, na.rm = TRUE),
            treated = first(treated), .groups = "drop")

if (length(unique(pre_baseline$treated)) < 2L) stop("Need treated & control regions pre-policy.")

n_treat <- sum(pre_baseline$tre

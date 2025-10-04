# --- CausalImpact pipeline (robust) ------------------------------------------
# Run from REPO ROOT:
#    Rscript scripts/causal_impact.R

need <- c("MatchIt","CausalImpact","tidyverse","ggplot2","zoo","broom")
new  <- setdiff(need, rownames(installed.packages()))
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
  library(MatchIt)
  library(CausalImpact)
  library(tidyverse)
  library(ggplot2)
  library(zoo)
  library(broom)
})

# --- Config ------------------------------------------------------------------
data_path   <- "data/ontario_cases.csv"
policy_date <- as.Date("2021-02-01")     # adjust if needed

# Output
fig_dir <- "figures"; dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
res_dir <- "results"; dir.create(res_dir,  showWarnings = FALSE, recursive = TRUE)
plot_path   <- file.path(fig_dir, "fig_causalimpact.png")
summary_txt <- file.path(res_dir, "causalimpact_summary.txt")

stopifnot(file.exists(data_path))
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Coerce literal "NA" strings to real NA, and ensure numerics are numeric
df[] <- lapply(df, function(x) if (is.character(x)) dplyr::na_if(x, "NA") else x)
num_cols <- c("incidence","treated")  # add more numeric cols if you have them
for (nm in intersect(num_cols, names(df))) {
  df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
}
df <- tidyr::drop_na(df) # Remove any rows containing NAs now that they are properly detected

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

# --- Matching on pre-policy baseline (ROBUST) ------------------------------
pre_df <- df %>% filter(week < policy_date)
if (nrow(pre_df) == 0L) stop("No pre-policy rows before ", policy_date) # Safeguard

pre_baseline <- pre_df %>%
  group_by(region) %>%
  summarise(
    mean_incidence = mean(incidence, na.rm = TRUE),
    treated = first(treated),
    .groups = "drop"
  )

if (length(unique(pre_baseline$treated)) < 2L) {
  stop("Need both treated and control regions pre-policy.")
}

n_treat <- sum(pre_baseline$treated == 1)
n_ctrl  <- sum(pre_baseline$treated == 0)
# Cap the ratio to a max of 5:1 for safety and prevent poor fit if n_ctrl is huge
match_ratio <- max(1L, min(5L, floor(n_ctrl / max(1L, n_treat))))  

m.out <- matchit(
  treated ~ mean_incidence,
  data   = pre_baseline,
  method = "nearest",
  ratio  = match_ratio,
  replace = TRUE                     # allow reuse of good controls
  # , caliper = 0.15 * sd(pre_baseline$mean_incidence, na.rm=TRUE) # uncomment to tighten matching
)

matched_data   <- match.data(m.out)
treated_regions <- matched_data %>% filter(treated == 1) %>% pull(region)
control_regions <- matched_data %>% filter(treated == 0) %>% pull(region)

# --- Controls: wide matrix with clean names (COVARIATES) --------------------
# Create the wide-format control series, ensuring clean column names (x_region)
# name controls as x_<region> to guarantee syntactic column names
control_wide <- df %>%
  filter(region %in% control_regions) %>%
  transmute(week, var = paste0("x_", as.character(region)), incidence) %>%
  tidyr::pivot_wider(names_from = var, values_from = incidence)


# --- Regular weekly index + fill -------------------------------------------
all_weeks <- tibble(week = seq(min(df$week, na.rm = TRUE),
                               max(df$week, na.rm = TRUE),
                               by = "week"))

treated_agg <- df %>%
  filter(region %in% treated_regions) %>%
  group_by(week) %>%
  summarise(y = mean(incidence, na.rm = TRUE), .groups = "drop")

df_ci <- all_weeks %>%
  left_join(treated_agg, by = "week") %>%
  left_join(control_wide, by = "week") %>%
  arrange(week) %>%
  # fill all non-week columns up/down to avoid NA breaks
  tidyr::fill(dplyr::everything(), .direction = "downup")

# Use a more verbose check for straddling the policy date
if (min(df_ci$week) >= policy_date || max(df_ci$week) <= policy_date) {
  stop("Series does not straddle policy_date; fix policy_date or data.")
}


# Select all data columns (y and all covariates) and convert to matrix
z_data <- df_ci %>% select(-week) %>% as.matrix()

# Convert to zoo object for CausalImpact
# z will now contain Y (treated mean) and multiple X covariates (matched control regions)
z <- zoo(z_data, order.by = df_ci$week)

# Define periods (using the full extent of the data for post-period)
pre_period  <- c(min(df_ci$week), policy_date - 7)
post_period <- c(policy_date,    max(df_ci$week))

impact <- CausalImpact(z, pre.period = pre_period, post.period = post_period,
                         model.args = list(nseasons = 52))

s <- capture.output({
  cat("=== CausalImpact summary ===\n")
  print(summary(impact))
  cat("\n=== CausalImpact report ===\n")
  print(summary(impact, "report"))
})
writeLines(s, summary_txt)

png(plot_path, width = 1400, height = 900, res = 150)
plot(impact)
dev.off()

cat("\nSaved:\n  ", summary_txt, "\n  ", plot_path, "\n\n")
try({
  ci_sum <- broom::tidy(impact$summary)
  print(head(ci_sum))
}, silent = TRUE)

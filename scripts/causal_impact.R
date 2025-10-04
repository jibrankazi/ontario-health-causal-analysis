# --- CausalImpact pipeline (robust) ------------------------------------------
# Run from REPO ROOT:
#     Rscript scripts/causal_impact.R

# Adding 'bsts', 'MatchIt', and 'jsonlite' for custom model building and results export
need <- c("MatchIt","CausalImpact","tidyverse","ggplot2","zoo","broom", "bsts", "jsonlite") 
new  <- setdiff(need, rownames(installed.packages()))
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
    library(MatchIt)
    library(CausalImpact)
    library(tidyverse)
    library(ggplot2)
    library(zoo)
    library(broom)
    library(bsts) # Explicitly load bsts for state specification functions
    library(jsonlite) # Required for writing compact JSON results
})

# --- Config ------------------------------------------------------------------
data_path   <- "data/ontario_cases.csv"
policy_date <- as.Date("2021-02-01")     # adjust if needed

# Output directories
fig_dir <- "figures"; dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
res_dir <- "results"; dir.create(res_dir,  showWarnings = FALSE, recursive = TRUE)
plot_path   <- file.path(fig_dir, "fig_causalimpact.png")
summary_txt <- file.path(res_dir, "causalimpact_summary.txt")

# --- Data Loading and Cleaning -----------------------------------------------
stopifnot(file.exists(data_path))
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Coerce literal "NA" strings to real NA, and ensure numerics are numeric
df[] <- lapply(df, function(x) if (is.character(x)) dplyr::na_if(x, "NA") else x)
num_cols <- c("incidence","treated")  # Add more numeric columns here if necessary
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

message(sprintf("Running matching with ratio %d:1", match_ratio))
m.out <- matchit(
    treated ~ mean_incidence,
    data    = pre_baseline,
    method = "nearest",
    ratio  = match_ratio,
    replace = TRUE         # allow reuse of good controls
)

matched_data    <- match.data(m.out)
treated_regions <- matched_data %>% filter(treated == 1) %>% pull(region) %>% unique()
control_regions <- matched_data %>% filter(treated == 0) %>% pull(region) %>% unique()

message("Matched treated regions: ", paste(treated_regions, collapse=", "))
message("Matched control regions: ", paste(control_regions, collapse=", "))

# --- Controls: wide matrix with clean names (COVARIATES) --------------------
# Create the wide-format control series, ensuring clean column names (x_region)
control_wide <- df %>%
    filter(region %in% control_regions) %>%
    # Use transmute for clean separation of columns used for wide format
    transmute(week, var = paste0("x_", as.character(region)), incidence) %>%
    tidyr::pivot_wider(names_from = var, values_from = incidence)


# --- Regular weekly index + fill -------------------------------------------
all_weeks <- tibble(week = seq(min(df$week, na.rm = TRUE),
                               max(df$week, na.rm = TRUE),
                               by = "week"))

# Aggregate treated regions (y series)
treated_agg <- df %>%
    filter(region %in% treated_regions) %>%
    group_by(week) %>%
    summarise(y = mean(incidence, na.rm = TRUE), .groups = "drop")

# Join series and fill missing data points (interpolation/extrapolation)
df_ci <- all_weeks %>%
    left_join(treated_agg, by = "week") %>%
    left_join(control_wide, by = "week") %>%
    arrange(week) %>%
    # fill all non-week columns up/down to avoid NA breaks for BSTS
    tidyr::fill(dplyr::everything(), .direction = "downup")

# Use a more verbose check for straddling the policy date
if (min(df_ci$week) >= policy_date || max(df_ci$week) <= policy_date) {
    stop("Series does not straddle policy_date; fix policy_date or data.")
}

# --- diagnostics: sample sizes & pre-period correlation with Y ---------------
pre_period  <- c(min(df_ci$week), policy_date - 7)
post_period <- c(policy_date,      max(df_ci$week))

n_pre  <- sum(df_ci$week <  policy_date)
n_post <- sum(df_ci$week >= policy_date)
message(sprintf("Pre points: %d | Post points: %d", n_pre, n_post))

# correlation of controls with treated series in PRE only
pre_mask <- df_ci$week < policy_date
y_pre <- df_ci$y[pre_mask]
X_pre <- df_ci[pre_mask, grepl("^x_", names(df_ci)), drop = FALSE]

# Ensure we have data to correlate
if (ncol(X_pre) > 0 && length(y_pre) > 1) {
    cors <- sapply(X_pre, function(v) suppressWarnings(cor(y_pre, v, use = "pairwise")))
    cors <- sort(cors, decreasing = TRUE)
    message("Top 10 pre-period correlations with Y:")
    print(head(cors, 10))
    
    # --- auto-select the best controls -----------------------------------------
    # Keep controls with reasonable pre-period correlation (tweak threshold as needed)
    keep <- names(cors)[which(abs(cors) >= 0.25)]  
    
    if (length(keep) < 2L && length(cors) >= 2L) {
        warning("Few informative controls (|r|>=0.25). Keeping the top 2 anyway.")
        keep <- names(cors)[seq_len(2)]
    } else if (length(keep) == 0L && length(cors) > 0) {
        warning("No informative controls found. Keeping the single best one.")
        keep <- names(cors)[1]
    }
    
    if (length(keep) > 0) {
        df_ci <- df_ci %>% dplyr::select(week, y, dplyr::all_of(keep))
    } else {
        message("No controls were selected after filtering.")
        df_ci <- df_ci %>% dplyr::select(week, y)
    }
} else {
    message("No control data or insufficient pre-period data for correlation analysis.")
    df_ci <- df_ci %>% dplyr::select(week, y)
}

# --- FINAL DATA CLEANUP AND PREP FOR BSTS ---
df_ci <- df_ci %>%
    # Explicitly coerce selected columns to numeric
    mutate(across(c(y, starts_with("x_")), as.numeric)) %>%
    # Drop any remaining rows that couldn't be filled/coerced
    tidyr::drop_na() 

# Final data for BSTS model
y <- df_ci$y
X <- as.matrix(df_ci %>% dplyr::select(-week, -y))

# GUARD: Ensure X is a matrix even if no columns were selected.
if (is.null(dim(X)) || ncol(X) == 0) {
    message("Proceeding with univariate BSTS (no covariates selected).")
    X <- matrix(numeric(0), nrow = length(y), ncol = 0)
}

# Use zoo with the Date index so CausalImpact knows the actual timeline
z_custom <- zoo::zoo(cbind(y, X), order.by = df_ci$week)


# --- Custom BSTS Model (Robust Priors/AR) --------------------------------
ss <- list()
# Tighter level prior helps shrink wandering trends
ss <- bsts::AddLocalLevel(ss, y, sigma.prior = bsts::SdPrior(0.05, sample.size = 32))
ss <- bsts::AddSeasonal(ss, y, nseasons = 52)    # weekly seasonality
# Modest AR helps when residual autocorr is visible
ss <- bsts::AddAutoAr(ss, y, lags = 1:2)

niter <- 5000  # raise for tighter posteriors if needed
fit <- bsts::bsts(y ~ X,
                  state.specification = ss,
                  niter = niter,
                  expected.model.size = min(15, ncol(X) + 1))


# --- Placebo Tests (optional, safe) ------------------------------------------
message("\nRunning Placebo Tests (checking pre-policy dates for false effects)...")
placebos <- as.Date(seq(policy_date - 140, policy_date - 14, by = "7 days"))
pvals <- numeric(0)
for (pd in placebos) {
  # Use df_ci$week and z_custom which are in scope
  pre_p  <- c(min(df_ci$week), pd - 7)
  post_p <- c(pd, max(df_ci$week))
  try({
    # Use z_custom for the data
    imp <- CausalImpact(z_custom, pre.period = pre_p, post.period = post_p,
                        model.args = list(nseasons = 52))
    s <- summary(imp)$summary
    if ("TailProb" %in% colnames(s) && "Average" %in% rownames(s)) {
      pvals <- c(pvals, as.numeric(s["Average","TailProb"]))
    }
  }, silent = TRUE)
}
if (length(pvals) == length(placebos)) {
  print(data.frame(placebo_date = placebos, p = pvals))
} else {
  message("Placebo p-values not available for all dates (",
          length(pvals), "/", length(placebos), "). Skipping table.")
}
message("--- End Placebo Tests ---\n")


impact <- CausalImpact(
    z_custom,
    pre.period  = pre_period,
    post.period = post_period,
    model.args  = list(bsts.model = fit)
)

# --- Save Summary and Plot ---------------------------------------------------
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

# --- Export compact JSON (for Python merge) ----------------------------------
# Pull the Posterior inference table safely
sum_tbl <- as.data.frame(summary(impact)$summary)

get_rc <- function(r, c) {
  if (r %in% rownames(sum_tbl) && c %in% colnames(sum_tbl)) {
    v <- sum_tbl[r, c]
    if (length(v) == 1 && !is.na(v)) return(as.numeric(v))
  }
  return(NA_real_)}
# ATT (Average) = Actual(Average) - Pred(Average)
att <- get_rc("Actual", "Average") - get_rc("Pred", "Average")
# 95% CI for ATT using Pred bounds:
# lower = Actual - Pred.upper; upper = Actual - Pred.lower
lo  <- get_rc("Actual", "Average") - get_rc("Pred.upper", "Average")
hi  <- get_rc("Actual", "Average") - get_rc("Pred.lower", "Average")
# Relative effect: may not be in the table; leave NA if absent
rel <- NA_real_
# p-value: not always present; try, else NA
p <- if ("TailProb" %in% colnames(sum_tbl) && "Average" %in% rownames(sum_tbl)) {
  as.numeric(sum_tbl["Average", "TailProb"])} else {
  NA_real_}

dir.create("results", showWarnings = FALSE, recursive = TRUE)
jsonlite::write_json(
  list(
    att = as.numeric(att),
    ci  = c(as.numeric(lo), as.numeric(hi)),
    p   = if (length(p) == 1) as.numeric(p) else NA_real_,
    relative_effect = if (is.na(rel)) NULL else as.numeric(rel),
    notes = NULL
  ),
  path = file.path("results", "bsts.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  na = "null"    # <-- ensures R NA becomes JSON null (not the string "NA")
)

cat("Saved: results/bsts.json\n")

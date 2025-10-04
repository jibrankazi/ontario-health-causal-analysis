# --- CausalImpact pipeline (robust) ------------------------------------------
# Run from REPO ROOT:
#     Rscript scripts/causal_impact.R

# Adding 'bsts' and 'jsonlite' for custom model building and results export
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
policy_date <- as.Date("2021-02-01")      # adjust if needed

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
    data    = pre_baseline,
    method = "nearest",
    ratio  = match_ratio,
    replace = TRUE           # allow reuse of good controls
    # , caliper = 0.15 * sd(pre_baseline$mean_incidence, na.rm=TRUE) # uncomment to tighten matching
)

matched_data    <- match.data(m.out)
treated_regions <- matched_data %>% filter(treated == 1) %>% pull(region)
control_regions <- matched_data %>% filter(treated == 0) %>% pull(region)

message("Matched treated regions: ", paste(unique(treated_regions), collapse=", "))
message("Matched control regions: ", paste(unique(control_regions), collapse=", "))

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

# --- Robustness Toggles ----------------------------------------------------
# (a) Log transform (handle zeros by +1)
# df_ci$y <- log1p(df_ci$y)
# for (nm in names(df_ci)[grepl("^x_", names(df_ci))]) df_ci[[nm]] <- log1p(df_ci[[nm]])

# (b) Winsorize top/bottom 1% (optional alternative to log)
# winsor <- function(v, p = 0.01) {
#    qs <- quantile(v, c(p, 1-p), na.rm = TRUE)
#    pmin(pmax(v, qs[[1]]), qs[[2]])}
# # df_ci$y <- winsor(df_ci$y)  # apply similarly to controls if needed

# --- diagnostics: sample sizes & pre-period correlation with Y ---------------
pre_period  <- c(min(df_ci$week), policy_date - 7)
post_period <- c(policy_date,      max(df_ci$week))

n_pre  <- sum(df_ci$week <  policy_date)
n_post <- sum(df_ci$week >= policy_date)
message(sprintf("Pre points: %d | Post points: %d", n_pre, n_post))
if (n_post < 8) warning("Post period has only ", n_post, " points; power will be limited.")

# correlation of controls with treated series in PRE only
pre_mask <- df_ci$week < policy_date
y_pre <- df_ci$y[pre_mask]
X_pre <- df_ci[pre_mask, grepl("^x_", names(df_ci)), drop = FALSE]
cors <- sapply(X_pre, function(v) suppressWarnings(cor(y_pre, v, use = "pairwise")))
cors <- sort(cors, decreasing = TRUE)
message("Top 10 pre-period correlations with Y:")
print(head(cors, 10))

# --- auto-select the best controls -----------------------------------------
keep <- names(cors)[which(abs(cors) >= 0.25)]  # tweak threshold
if (length(keep) < 2L) {
    warning("Few informative controls (|r|>=0.25). Keeping the top 2 anyway.")
    keep <- names(cors)[seq_len(min(2, length(cors)))]
}

df_ci <- df_ci %>%
    dplyr::select(week, y, dplyr::all_of(keep))

# --- PCA (Optional) Compression --------------------------------------------
# If the remaining number of controls is still large (>5), you can compress them:
# X_final <- as.matrix(df_ci %>% dplyr::select(-week, -y))
# pc <- prcomp(X_final, center = TRUE, scale. = TRUE)
# k <- min(5, ncol(X_final))
# z_data <- cbind(y = df_ci$y, pc$x[, 1:k, drop = FALSE])
# colnames(z_data)[1] <- "y"
# z <- zoo::zoo(z_data, order.by = df_ci$week)
# NOTE: If you use PCA, comment out the next block and uncomment the PCA block above.


# --- FINAL DATA CLEANUP AND PREP FOR BSTS ---
# Ensures all selected columns are purely numeric and contain no NAs, 
# which is crucial for BSTS's matrix input and fixes the "string to float: 'NA'" error.
df_ci <- df_ci %>%
    # Explicitly coerce selected columns to numeric
    mutate(across(c(y, starts_with("x_")), as.numeric)) %>%
    # Drop any remaining rows that couldn't be filled/coerced (should be rare)
    tidyr::drop_na() 

# Final data for BSTS model
z_data <- df_ci %>% dplyr::select(-week) %>% as.matrix()
z <- zoo::zoo(z_data, order.by = df_ci$week)


# --- Placebo Tests (Falsification Check) -----------------------------------
message("\nRunning Placebo Tests (checking pre-policy dates for false effects)...")
placebos <- as.Date(seq(policy_date - 140, policy_date - 14, by = "7 days"))
pvals <- c()
for (pd in placebos) {
    # Use whole weeks for pre/post split
    pre_p  <- c(min(df_ci$week), pd - 7)
    post_p <- c(pd, max(df_ci$week))
    try({
        # Use the simple CausalImpact call for speed in placebos
        imp <- CausalImpact(z, pre.period = pre_p, post.period = post_p, model.args = list(nseasons = 52))
        pvals <- c(pvals, summary(imp)$summary$TailProb[2])  # "Average" row
    }, silent = TRUE)
}
print(data.frame(placebo_date = placebos, p = pvals))
message("--- End Placebo Tests ---\n")


# --- Custom BSTS Model (Robust Priors/AR) --------------------------------
y <- df_ci$y
X <- as.matrix(df_ci %>% dplyr::select(-week, -y))

# GUARD: Ensure X is a matrix even if no columns were selected.
if (is.null(dim(X)) || ncol(X) == 0) {
    message("No controls after selection; proceeding with univariate BSTS.")
    # keep X as a zero-column matrix so cbind(y, X) still works:
    X <- matrix(numeric(0), nrow = length(y), ncol = 0)
}

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

# Use zoo with the Date index so CausalImpact knows the actual timeline
z_custom <- zoo::zoo(cbind(y, X), order.by = df_ci$week)

impact <- CausalImpact(
    z_custom,
    pre.period  = pre_period,
    post.period = post_period,
    model.args  = list(bsts.model = fit)
)

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
sum_tbl <- summary(impact)$summary

get_cell <- function(row_name, col_name) {
  if (row_name %in% rownames(sum_tbl) && col_name %in% colnames(sum_tbl)) {
    as.numeric(sum_tbl[row_name, col_name])
  } else {
    NA_real_
  }}# Rows are "Average" / "Cumulative"; columns are metric names.
att <- get_cell("Average", "AbsEffect")
lo  <- get_cell("Average", "AbsEffect.lower")
hi  <- get_cell("Average", "AbsEffect.upper")
rel <- get_cell("Average", "RelEffect")
if (!is.na(rel)) rel <- rel / 100  # convert % → proportion
# p-value column is usually "TailProb"; fall back to "p" if present
p <- if ("TailProb" %in% colnames(sum_tbl)) {
  get_cell("Average", "TailProb")} else if ("p" %in% colnames(sum_tbl)) {
  get_cell("Average", "p")} else {
  NA_real_}

dir.create("results", showWarnings = FALSE, recursive = TRUE)
jsonlite::write_json(
  list(att = att, ci = c(lo, hi), p = p, relative_effect = rel, notes = NULL),
  path = file.path("results", "bsts.json"),
  auto_unbox = TRUE, pretty = TRUE, null = "null")
cat("Saved: results/bsts.json\n")

try({
    ci_sum <- broom::tidy(impact$summary)
    print(head(ci_sum))
}, silent = TRUE)

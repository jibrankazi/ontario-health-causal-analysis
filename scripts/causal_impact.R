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


# --- Helper Function: Safely extract numeric value from a table ---
# This function is now defensive, allowing the caller to decide whether to 
# warn, fail, or silently return NA_real_ on missing keys or failed coercion.
get_rc <- function(tbl, r, c, warn = TRUE, fail_on_missing = FALSE) {
    # 1. Input Type Check
    if (!is.matrix(tbl) && !is.data.frame(tbl)) {
        stop("tbl must be a matrix or data.frame.")
    }
    
    # 2. Check for row and column existence
    missing_row <- !(r %in% rownames(tbl))
    missing_col <- !(c %in% colnames(tbl))
    
    if (missing_row || missing_col) {
        msg <- paste0(
            if (missing_row) sprintf("Row '%s' not found. ", r) else "",
            if (missing_col) sprintf("Column '%s' not found.", c) else ""
        )
        if (fail_on_missing) stop(msg)
        if (warn) warning(msg)
        return(NA_real_)
    }
    
    # 3. Extract and coerce to numeric (safely)
    # Use drop=TRUE to ensure a scalar is returned for a single cell extraction.
    val <- as.numeric(tbl[r, c, drop = TRUE])
    
    # 4. Check for failed coercion and warn if requested
    # as.numeric returns NA if the value cannot be coerced.
    if (is.na(val) && warn) {
        warning(sprintf("Value at [%s, %s] could not be coerced to numeric or is NA.", r, c))
    }
    
    # Returns the value (which will be NA_real_ if coercion failed)
    return(val) 
}
# --- End Helper Function --------------------------------------------------

# --- Parsing Helpers for CausalImpact Summary -------------------------------
parse_point <- function(x, remove_pct = FALSE) {
    if (is.na(x) || is.null(x)) return(NA_real_)
    val_str <- as.character(x)
    # Remove (s.d.) part
    if (grepl("\\(", val_str)) {
        val_str <- strsplit(val_str, " \\(")[[1]][1]
    }
    # Remove % if requested
    if (remove_pct) val_str <- gsub("%", "", val_str)
    as.numeric(val_str)
}

parse_ci_bounds <- function(x) {
    if (is.na(x) || is.null(x)) return(c(NA_real_, NA_real_))
    val_str <- gsub("\\[|\\]", "", as.character(x))
    parts <- strsplit(val_str, ",")[[1]]
    lower_str <- gsub("%", "", trimws(parts[1]))
    upper_str <- gsub("%", "", trimws(parts[2]))
    c(as.numeric(lower_str), as.numeric(upper_str))
}
# --- End Parsing Helpers --------------------------------------------------


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


# --- Initialize result variables for robustness
att <- NA_real_
lo  <- NA_real_
hi  <- NA_real_
rel <- NA_real_
p   <- NA_real_
bsts_reason <- NULL

# --- Model Fitting and Analysis (Wrapped in tryCatch) ------------------------
tryCatch({
    
    # --- Run Main Causal Impact Analysis ---
    # FIX: Reverting to internal model building to avoid "illegal extra args: 'bsts.model'"
    # We still specify seasonality via model.args.
    impact <- CausalImpact(
        z_custom,
        pre.period  = pre_period,
        post.period = post_period,
        model.args  = list(nseasons = 52) 
    )
    
    # --- Save Summary and Plot (only on success) --------------------------------
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

    # --- Extract Compact JSON Results (using local assignment) -----------------
    sum_mat <- impact$summary
    
    # Find CI rows
    ci_rows <- which(rownames(sum_mat) == "95% CI")
    if (length(ci_rows) < 2) stop("Unexpected structure in summary matrix.")
    pred_ci_row <- ci_rows[1]
    abs_ci_row <- ci_rows[2]
    
    # Extract values using parsers
    actual_avg <- parse_point(sum_mat["Actual", "Average"])
    pred_avg <- parse_point(sum_mat["Prediction (s.d.)", "Average"])
    att <- parse_point(sum_mat["Absolute effect (s.d.)", "Average"])
    rel_val <- parse_point(sum_mat["Relative effect (s.d.)", "Average"], remove_pct = TRUE) / 100
    
    # CIs for absolute effect
    abs_ci <- parse_ci_bounds(sum_mat[abs_ci_row, "Average"])
    lo <- abs_ci[1]
    hi <- abs_ci[2]
    
    # P-value
    p <- impact$inference$TailProb
    
    # Local assignments (<-) to variables initialized outside the tryCatch
    # (already done above)
    
}, error = function(e) {
    # On error, use global assignment (<<-) only for the reason variable, 
    # ensuring it's captured in the parent scope.
    bsts_reason <<- paste0("BSTS via Rscript failed: ", e$message)
    message(bsts_reason)
    
    # --- Simplify error handling for file cleanup ---
    # Removes files only if they exist
    files_to_check <- c(summary_txt, plot_path)
    file.remove(files_to_check[file.exists(files_to_check)])
})

# --- Export compact JSON (runs always, using initialized/calculated values) ---
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# Ensure the final output uses the calculated 'att', 'lo', 'hi', 'rel', 'p' (which 
# will be NA_real_ if the tryCatch block failed) and the captured 'bsts_reason'.
jsonlite::write_json(
  list(
    att = as.numeric(att),
    ci  = c(as.numeric(lo), as.numeric(hi)),
    p   = if (length(p) == 1 && is.finite(p)) as.numeric(p) else NA_real_,
    relative_effect = if (!is.na(rel)) as.numeric(rel) else NA_real_,
    notes = NULL,
    bsts_reason = bsts_reason # Include the error reason here
  ),
  path = file.path("results", "bsts.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  na = "null"    # <-- ensures R NA/NaN becomes JSON null (required for robustness)
)

cat("Saved: results/bsts.json\n")

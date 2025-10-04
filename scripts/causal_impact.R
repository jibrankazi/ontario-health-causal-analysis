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
    library(bsts)
    library(jsonlite)
})

# --- Config ------------------------------------------------------------------
data_path   <- "data/ontario_cases.csv"
policy_date <- as.Date("2021-02-01")

# Output directories
fig_dir <- "figures"; dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
res_dir <- "results"; dir.create(res_dir,  showWarnings = FALSE, recursive = TRUE)
plot_path   <- file.path(fig_dir, "fig_causalimpact.png")
summary_txt <- file.path(res_dir, "causalimpact_summary.txt")


# --- Helper Function: Safely extract numeric value from a table ---
get_rc <- function(tbl, r, c, warn = TRUE, fail_on_missing = FALSE) {
    if (!is.matrix(tbl) && !is.data.frame(tbl)) {
        stop("tbl must be a matrix or data.frame.")
    }
    
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
    
    val <- as.numeric(tbl[r, c, drop = TRUE])
    
    if (is.na(val) && warn) {
        warning(sprintf("Value at [%s, %s] could not be coerced to numeric or is NA.", r, c))
    }
    
    return(val) 
}

# --- Parsing Helpers for CausalImpact Summary -------------------------------
parse_point <- function(x, remove_pct = FALSE) {
    if (is.na(x) || is.null(x)) return(NA_real_)
    if (is.numeric(x)) return(x)
    val_str <- trimws(as.character(x))
    if (grepl("\\(", val_str)) {
        val_str <- strsplit(val_str, "\\s*\\(")[[1]][1]
        val_str <- trimws(val_str)
    }
    if (remove_pct) val_str <- gsub("%", "", val_str)
    val_str <- trimws(val_str)
    num_val <- suppressWarnings(as.numeric(val_str))
    if (is.na(num_val)) return(NA_real_)
    return(num_val)
}

parse_ci_bounds <- function(x) {
    if (is.na(x) || is.null(x)) return(c(NA_real_, NA_real_))
    if (is.numeric(x)) return(c(x - 1.96 * sd(x), x + 1.96 * sd(x)))
    val_str <- trimws(gsub("\\[|\\]", "", as.character(x)))
    if (val_str == "") return(c(NA_real_, NA_real_))
    parts <- strsplit(val_str, ",")[[1]]
    if (length(parts) != 2) return(c(NA_real_, NA_real_))
    lower_str <- trimws(gsub("%", "", parts[1]))
    upper_str <- trimws(gsub("%", "", parts[2]))
    lower <- suppressWarnings(as.numeric(lower_str))
    upper <- suppressWarnings(as.numeric(upper_str))
    c(lower, upper)
}

# --- Data Loading and Cleaning -----------------------------------------------
stopifnot(file.exists(data_path))
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Coerce literal "NA" strings to real NA, and ensure numerics are numeric
df[] <- lapply(df, function(x) if (is.character(x)) dplyr::na_if(x, "NA") else x)
num_cols <- c("incidence","treated")
for (nm in intersect(num_cols, names(df))) {
    df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
}
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

# --- Matching on pre-policy baseline (ROBUST) ------------------------------
pre_df <- df %>% filter(week < policy_date)
if (nrow(pre_df) == 0L) stop("No pre-policy rows before ", policy_date)

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
# Use adaptive ratio: minimum of 3:1, but don't exceed available controls
match_ratio <- max(1L, min(3L, floor(n_ctrl / max(1L, n_treat))))

message(sprintf("Running matching with ratio %d:1", match_ratio))
m.out <- matchit(
    treated ~ mean_incidence,
    data    = pre_baseline,
    method = "nearest",
    ratio  = match_ratio,
    replace = TRUE
)

matched_data    <- match.data(m.out)
treated_regions <- matched_data %>% filter(treated == 1) %>% pull(region) %>% unique()
control_regions <- matched_data %>% filter(treated == 0) %>% pull(region) %>% unique()

message("Matched treated regions: ", paste(treated_regions, collapse=", "))
message("Matched control regions: ", paste(control_regions, collapse=", "))

# --- Controls: wide matrix with clean names (COVARIATES) --------------------
control_wide <- df %>%
    filter(region %in% control_regions) %>%
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

# Join series and fill missing data points
df_ci <- all_weeks %>%
    left_join(treated_agg, by = "week") %>%
    left_join(control_wide, by = "week") %>%
    arrange(week) %>%
    tidyr::fill(dplyr::everything(), .direction = "downup")

# Validate date range
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

if (ncol(X_pre) > 0 && length(y_pre) > 1) {
    cors <- sapply(X_pre, function(v) suppressWarnings(cor(y_pre, v, use = "pairwise")))
    cors <- sort(cors, decreasing = TRUE)
    message("Top 10 pre-period correlations with Y:")
    print(head(cors, 10))
    
    # Auto-select the best controls
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
    mutate(across(c(y, starts_with("x_")), ~ suppressWarnings(as.numeric(.x)))) %>%
    tidyr::drop_na() 

# Final data for BSTS model
y <- df_ci$y
X <- as.matrix(df_ci %>% dplyr::select(-week, -y))

if (is.null(dim(X)) || ncol(X) == 0) {
    message("Proceeding with univariate BSTS (no covariates selected).")
    X <- matrix(numeric(0), nrow = length(y), ncol = 0)
}

# Use zoo with the Date index
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
    
    # Run Main Causal Impact Analysis
    impact <- CausalImpact(
        z_custom,
        pre.period  = pre_period,
        post.period = post_period,
        model.args  = list(nseasons = 52) 
    )
    
    # Save Summary and Plot
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

    # Extract Compact JSON Results
    sum_mat <- impact$summary
    
    rownames(sum_mat) <- trimws(rownames(sum_mat))
    colnames(sum_mat) <- trimws(colnames(sum_mat))
    
    abs_row <- which(grepl("Absolute effect", rownames(sum_mat)))
    if (length(abs_row) == 0) {
        stop("Missing 'Absolute effect' row in summary")
    }
    abs_row <- abs_row[1]
    
    actual_row <- which(grepl("Actual", rownames(sum_mat)))
    pred_row <- which(grepl("Prediction", rownames(sum_mat)))
    rel_row <- which(grepl("Relative effect", rownames(sum_mat)))
    
    actual_avg <- if (length(actual_row) > 0) parse_point(sum_mat[actual_row[1], "Average"]) else NA_real_
    pred_avg <- if (length(pred_row) > 0) parse_point(sum_mat[pred_row[1], "Average"]) else NA_real_
    att <- parse_point(sum_mat[abs_row, "Average"])
    if (length(rel_row) > 0) {
        rel_pct <- parse_point(sum_mat[rel_row[1], "Average"], remove_pct = TRUE)
        rel <- if (!is.na(rel_pct)) rel_pct / 100 else NA_real_
    } else {
        rel <- NA_real_
    }
    
    abs_ci_row <- abs_row + 1
    if (abs_ci_row <= nrow(sum_mat) && grepl("95% CI", rownames(sum_mat)[abs_ci_row])) {
        abs_ci <- parse_ci_bounds(sum_mat[abs_ci_row, "Average"])
        lo <- abs_ci[1]
        hi <- abs_ci[2]
    } else {
        lo <- NA_real_
        hi <- NA_real_
    }
    
    p <- impact$inference$TailProb
    
    # Robust fallback for ATT
    if (is.na(att) && !is.na(actual_avg) && !is.na(pred_avg)) {
        att <- actual_avg - pred_avg
    }
    
    cat("Extracted: att=", att, " lo=", lo, " hi=", hi, " rel=", rel, " p=", p, "\n")
    
}, error = function(e) {
    bsts_reason <- paste0("BSTS via Rscript failed: ", e$message)
    message(bsts_reason)
    
    # Store reason in parent environment
    assign("bsts_reason", bsts_reason, envir = parent.frame())
    assign("att", NA_real_, envir = parent.frame())
    assign("lo", NA_real_, envir = parent.frame())
    assign("hi", NA_real_, envir = parent.frame())
    assign("rel", NA_real_, envir = parent.frame())
    assign("p", NA_real_, envir = parent.frame())
    
    files_to_check <- c(summary_txt, plot_path)
    invisible(file.remove(files_to_check[file.exists(files_to_check)]))
})

# --- Export compact JSON ---
dir.create("results", showWarnings = FALSE, recursive = TRUE)

jsonlite::write_json(
  list(
    att = if (is.na(att)) NULL else att,
    ci  = c(if (is.na(lo)) NULL else lo, if (is.na(hi)) NULL else hi),
    p   = if (length(p) == 1 && is.finite(p)) p else NULL,
    relative_effect = if (is.na(rel)) NULL else rel,
    notes = NULL,
    bsts_reason = bsts_reason
  ),
  path = file.path("results", "bsts.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  na = "null"
)

cat("Saved: results/bsts.json\n")

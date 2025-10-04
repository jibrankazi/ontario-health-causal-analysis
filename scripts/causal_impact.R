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

n_treat <- sum(pre_baseline$treated == 1)
n_ctrl  <- sum(pre_baseline$treated == 0)
match_ratio <- max(1L, min(5L, floor(n_ctrl / max(1L, n_treat))))
message(sprintf("Running matching with ratio %d:1", match_ratio))

m.out <- matchit(treated ~ mean_incidence,
                 data = pre_baseline,
                 method = "nearest",
                 ratio = match_ratio,
                 replace = TRUE)

matched <- match.data(m.out)
treated_regions <- unique(matched$region[matched$treated == 1])
control_regions <- unique(matched$region[matched$treated == 0])

message("Matched treated regions: ", paste(treated_regions, collapse=", "))
message("Matched control regions: ", paste(control_regions, collapse=", "))

# --- Build series ------------------------------------------------------------
all_weeks <- tibble(week = seq(min(df$week, na.rm = TRUE),
                               max(df$week, na.rm = TRUE),
                               by = "week"))

treated_agg <- df %>%
  filter(region %in% treated_regions) %>%
  group_by(week) %>%
  summarize(y = mean(incidence, na.rm = TRUE), .groups = "drop")

control_wide <- df %>%
  filter(region %in% control_regions) %>%
  transmute(week, var = paste0("x_", as.character(region)), incidence) %>%
  tidyr::pivot_wider(names_from = var, values_from = incidence)

df_ci <- all_weeks %>%
  left_join(treated_agg, by = "week") %>%
  left_join(control_wide, by = "week") %>%
  arrange(week) %>%
  tidyr::fill(dplyr::everything(), .direction = "downup")

if (min(df_ci$week) >= policy_date || max(df_ci$week) <= policy_date)
  stop("Series does not straddle policy_date; fix policy_date or data.")

# diagnostics + auto-select controls by pre correlation
pre_period  <- c(min(df_ci$week), policy_date - 7)
post_period <- c(policy_date,      max(df_ci$week))

n_pre  <- sum(df_ci$week <  policy_date)
n_post <- sum(df_ci$week >= policy_date)
message(sprintf("Pre points: %d | Post points: %d", n_pre, n_post))

pre_mask <- df_ci$week < policy_date
y_pre <- df_ci$y[pre_mask]
X_pre <- df_ci[pre_mask, grepl("^x_", names(df_ci)), drop = FALSE]

if (ncol(X_pre) > 0 && length(y_pre) > 1) {
  cors <- sapply(X_pre, function(v) suppressWarnings(cor(y_pre, v, use = "pairwise")))
  cors <- sort(cors, decreasing = TRUE)
  message("Top 10 pre-period correlations with Y:")
  print(head(cors, 10))

  keep <- names(cors)[which(abs(cors) >= 0.25)]
  if (length(keep) < 2L && length(cors) >= 2L) keep <- names(cors)[1:2]
  if (length(keep) == 0L && length(cors) > 0)   keep <- names(cors)[1]
  if (length(keep) > 0) df_ci <- df_ci %>% dplyr::select(week, y, dplyr::all_of(keep))
} else {
  message("No control data or insufficient pre-period data; proceeding univariate.")
  df_ci <- df_ci %>% dplyr::select(week, y)
}

# final cleanup
df_ci <- df_ci %>% mutate(across(c(y, starts_with("x_")), as.numeric)) %>% tidyr::drop_na()
y <- df_ci$y
X <- as.matrix(df_ci %>% dplyr::select(-week, -y))
if (is.null(dim(X)) || ncol(X) == 0) X <- matrix(numeric(0), nrow = length(y), ncol = 0)

z_custom <- zoo::zoo(cbind(y, X), order.by = df_ci$week)

# --- BSTS model (version-safe prior) ----------------------------------------
ss <- list()

# Guard: SdPrior is not exported in some bsts versions
if ("SdPrior" %in% getNamespaceExports("bsts")) {
  ss <- bsts::AddLocalLevel(ss, y, sigma.prior = bsts::SdPrior(0.05, sample.size = 32))
} else {
  ss <- bsts::AddLocalLevel(ss, y)  # fallback to default prior
}
ss <- bsts::AddSeasonal(ss, y, nseasons = 52)
ss <- bsts::AddAutoAr(ss, y, lags = 1:2)

fit <- bsts::bsts(y ~ X, state.specification = ss,
                  niter = 5000,
                  expected.model.size = min(15, ncol(X) + 1))

impact <- CausalImpact(z_custom,
                       pre.period  = pre_period,
                       post.period = post_period,
                       model.args  = list(bsts.model = fit))

# --- Save summary + plot -----------------------------------------------------
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

# --- Compact JSON (numeric p or null) ---------------------------------------
sum_tbl <- as.data.frame(summary(impact)$summary)

get_rc <- function(r, c) {
  if (r %in% rownames(sum_tbl) && c %in% colnames(sum_tbl)) {
    v <- sum_tbl[r, c]; if (length(v)==1 && !is.na(v)) return(as.numeric(v))
  }
  return(NA_real_)
}

att <- get_rc("Actual","Average") - get_rc("Pred","Average")
lo  <- get_rc("Actual","Average") - get_rc("Pred.upper","Average")
hi  <- get_rc("Actual","Average") - get_rc("Pred.lower","Average")
p   <- if ("TailProb" %in% colnames(sum_tbl) && "Average" %in% rownames(sum_tbl))
         as.numeric(sum_tbl["Average","TailProb"]) else NA_real_
if (!is.finite(p)) p <- NA_real_

jsonlite::write_json(
  list(
    att = as.numeric(att),
    ci  = c(as.numeric(lo), as.numeric(hi)),
    p   = if (length(p)==1) as.numeric(p) else NA_real_,
    relative_effect = NULL,
    notes = NULL
  ),
  path = file.path("results","bsts.json"),
  auto_unbox = TRUE,
  pretty     = TRUE,
  null       = "null",
  na         = "null"
)
cat("Saved: results/bsts.json\n")

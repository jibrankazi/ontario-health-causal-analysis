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

# Match on pre-policy average incidence (by region)
pre_df <- df %>% filter(week < policy_date)
if (nrow(pre_df) == 0L) stop("No pre-policy rows before ", policy_date)

pre_baseline <- pre_df %>%
  group_by(region) %>%
  summarize(
    mean_incidence = mean(incidence, na.rm = TRUE),
    treated = first(treated),
    .groups = "drop"
  )
if (length(unique(pre_baseline$treated)) < 2L) {
  stop("Need both treated and control regions pre-policy.")
}

m.out <- matchit(treated ~ mean_incidence,
                 data  = pre_baseline,
                 method = "nearest")
matched_regions <- match.data(m.out)$region

panel <- df %>% filter(region %in% matched_regions)

# Aggregate weekly treated/control means
agg <- panel %>%
  group_by(week) %>%
  summarize(
    treated_mean = mean(incidence[treated == 1], na.rm=TRUE),
    control_mean = mean(incidence[treated == 0], na.rm=TRUE),
    .groups = "drop"
  ) %>%
  arrange(week)

# Fill any missing weeks and carry forward
all_weeks <- tibble(week = seq(min(agg$week), max(agg$week), by = "week"))
agg <- all_weeks %>%
  left_join(agg, by = "week") %>%
  tidyr::fill(treated_mean, control_mean, .direction = "downup")

if (min(agg$week) >= policy_date || max(agg$week) <= policy_date) {
  stop("Series does not straddle policy_date; fix policy_date or data.")
}

z <- zoo(cbind(y = agg$treated_mean, x1 = agg$control_mean), order.by = agg$week)

pre_period  <- c(min(agg$week), policy_date - 7)
post_period <- c(policy_date,    max(agg$week))

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

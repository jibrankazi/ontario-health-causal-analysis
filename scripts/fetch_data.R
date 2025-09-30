#!/usr/bin/env Rscript
# Fetch public resources from the Ontario Data Catalogue (CKAN).

suppressPackageStartupMessages({
  library(httr); library(jsonlite); library(readr); library(dplyr)})

dir.create("data", showWarnings = FALSE, recursive = TRUE)

# --- EXAMPLE CODE (UNCOMMENT & EDIT) ---
# pkg <- fromJSON(content(GET("https://data.ontario.ca/api/3/action/package_show?id=<DATASET_ID>"), as="text"))$result
# res <- pkg$resources[[1]]$url  # choose the CSV resource you need
# message("Downloading: ", res)
# dat <- read_csv(res, show_col_types = FALSE)
# write_csv(dat, "data/ontario_incidence.csv")

message("Edit scripts/fetch_data.R with the correct dataset IDs/resources, then run: Rscript scripts/fetch_data.R")

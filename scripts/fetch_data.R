#!/usr/bin/env Rscript
# Fill in the correct dataset/resource IDs from https://data.ontario.ca/
suppressPackageStartupMessages({ library(httr); library(jsonlite); library(readr); library(dplyr) })
dir.create("data", showWarnings = FALSE, recursive = TRUE)

# Example CKAN flow:
# pkg <- fromJSON(content(GET("https://data.ontario.ca/api/3/action/package_show?id=<DATASET_ID>"), as="text"))$result
# res <- pkg$resources[[1]]$url  # pick the CSV you need
# message("Downloading: ", res)
# WRITE THE DATA TO THE CORRECT FILENAME EXPECTED BY causal_impact.R
# write_csv(read_csv(res, show_col_types = FALSE), "data/ontario_cases.csv")

message("Edit scripts/fetch_data.R with the correct dataset IDs/resources, then run: Rscript scripts/fetch_data.R")
message("NOTE: The file name has been corrected to 'data/ontario_cases.csv'.")

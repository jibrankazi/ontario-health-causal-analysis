#!/usr/bin/env Rscript
# Fetch Ontario COVID-19 case data from data.ontario.ca
# This is a working template - update dataset IDs as needed

suppressPackageStartupMessages({
    library(httr)
    library(jsonlite)
    library(readr)
    library(dplyr)
})

dir.create("data", showWarnings = FALSE, recursive = TRUE)

# Configuration - UPDATE THESE VALUES
# Find dataset IDs at https://data.ontario.ca/
DATASET_ID <- "your-dataset-id-here"  # e.g., "confirmed-positive-cases-of-covid-19-in-ontario"
RESOURCE_IDX <- 1  # Which resource in the dataset to use (usually 1)

# Ontario Open Data API endpoint
API_BASE <- "https://data.ontario.ca/api/3/action"

# Function to fetch dataset metadata
fetch_dataset_info <- function(dataset_id) {
    url <- paste0(API_BASE, "/package_show?id=", dataset_id)
    response <- GET(url)
    
    if (status_code(response) != 200) {
        stop("Failed to fetch dataset info. Status: ", status_code(response))
    }
    
    content(response, as = "parsed", type = "application/json")
}

# Function to download and save data
download_data <- function(dataset_id, resource_idx = 1, output_path = "data/ontario_cases.csv") {
    message("Fetching dataset metadata for: ", dataset_id)
    
    # Get dataset info
    pkg_info <- fetch_dataset_info(dataset_id)
    
    if (!pkg_info$success) {
        stop("API request failed: ", pkg_info$error$message)
    }
    
    # Get resources list
    resources <- pkg_info$result$resources
    
    if (length(resources) == 0) {
        stop("No resources found in dataset")
    }
    
    if (resource_idx > length(resources)) {
        stop("Resource index ", resource_idx, " out of bounds. Dataset has ", 
             length(resources), " resources.")
    }
    
    # Get the specified resource
    resource <- resources[[resource_idx]]
    resource_url <- resource$url
    resource_format <- tolower(resource$format)
    
    message("Downloading resource: ", resource$name)
    message("Format: ", resource_format)
    message("URL: ", resource_url)
    
    # Download based on format
    if (resource_format == "csv") {
        data <- read_csv(resource_url, show_col_types = FALSE)
    } else if (resource_format %in% c("json", "geojson")) {
        data <- fromJSON(resource_url)
        if (is.list(data) && "features" %in% names(data)) {
            # Handle GeoJSON
            data <- data$features$properties
        }
        data <- as_tibble(data)
    } else {
        warning("Unsupported format: ", resource_format, ". Attempting CSV read...")
        data <- read_csv(resource_url, show_col_types = FALSE)
    }
    
    # Save to file
    write_csv(data, output_path)
    message("Data saved to: ", output_path)
    message("Rows: ", nrow(data), " | Columns: ", ncol(data))
    
    # Show column names
    message("\nColumn names:")
    print(names(data))
    
    return(data)
}

# Main execution
tryCatch({
    if (DATASET_ID == "your-dataset-id-here") {
        message("=" , rep("=", 70))
        message("SETUP REQUIRED")
        message("=" , rep("=", 70))
        message("\nPlease edit scripts/fetch_data.R with the correct dataset ID.")
        message("\nSteps:")
        message("1. Visit https://data.ontario.ca/")
        message("2. Find your dataset (e.g., COVID-19 cases)")
        message("3. Copy the dataset ID from the URL")
        message("4. Update DATASET_ID in this script")
        message("5. Run: Rscript scripts/fetch_data.R")
        message("\nExample dataset IDs:")
        message("  - confirmed-positive-cases-of-covid-19-in-ontario")
        message("  - status-of-covid-19-cases-in-ontario")
        message("\n")
    } else {
        # Actually download the data
        data <- download_data(DATASET_ID, RESOURCE_IDX)
        message("\n✓ Data successfully downloaded and saved!")
    }
}, error = function(e) {
    message("\n✗ Error: ", e$message)
    message("\nTroubleshooting:")
    message("  - Verify the dataset ID is correct")
    message("  - Check your internet connection")
    message("  - Ensure the dataset is publicly accessible")
    quit(status = 1)
})

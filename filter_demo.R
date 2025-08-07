# ---------------------------------------------------------------
# Shark Age Estimation Demo Script
# ---------------------------------------------------------------

# Load required libraries
library(ggplot2)     # For plotting

# Get the directory of the current script file then set the working directory.
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    # If running in RStudio
    current_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
    setwd(current_dir)
} else {
    # Fallback for command line execution
    path <- dirname(sys.frame(1)$ofile)
    setwd(path)
}

# Source the custom filtering functions from filter_utils.R
source("filter_utils.R")

# Define the directory where the data is stored
data_dir <- "./data/"
data_fn <- paste0(data_dir, "cleaned_data.txt")

# Read the cleaned data into a data.frame
all_elements <- read.table(data_fn, header = TRUE, sep = ",", stringsAsFactors = FALSE)
# Print the first few rows to verify the data loaded correctly
print(head(all_elements))

# Choose the element/column to analyze
elt_name <- "S34_ppm"

# Specify the cases (individual sharks) to analyze
cases <- c(43, 44, 55, 56)

# For each case, extract the data for the selected element
elt_array <- list()
for (i in seq_along(cases)) {
    case_id <- cases[i]
    elt_array[[i]] <- all_elements[all_elements$case == case_id, elt_name]
}

# Set parameters for the age estimation algorithm
bandwidth <- 30
threshold <- 2
criterion <- "aic"

# Initialize lists to store plots and results for each case
plts <- list()

# Loop over each case's data and apply the age estimation algorithm
for (i in seq_along(elt_array)) {
    # Convert the selected element data to numeric (in case it's not already)
    elt <- as.numeric(elt_array[[i]])
    
    # Apply the age estimation algorithm in the "forward" direction
    # This tries to estimate the number of age markers (e.g., growth rings) from start to end
    fwd <- age_shark(elt, bandwidth, "forward", threshold, criterion)
    # Apply the age estimation algorithm in the "backward" direction (end to start)
    bkwd <- age_shark(elt, bandwidth, "backward", threshold, criterion)

    # Extract the estimated age (number of detected markers) and their positions for both directions
    fwd_count <- fwd$age
    fwd_locations <- fwd$peak_indices
    fwd_smoothed <- fwd$fitted

    bkwd_count <- bkwd$age
    bkwd_locations <- bkwd$peak_indices
    bkwd_smoothed <- bkwd$fitted

    # Create a data frame to hold the raw and smoothed data for plotting
    df <- data.frame(
        idx = seq_along(elt),   # Index along the sample (e.g., distance along a shark vertebra)
        raw = elt,              # Raw measured values (e.g., chemical concentration)
        fwd = fwd_smoothed,     # Smoothed values from forward estimation
        bkwd = bkwd_smoothed    # Smoothed values from backward estimation
    )

    # Plot the raw and smoothed data
    # - Grey line: raw data
    # - Blue line: smoothed data (forward)
    # - Red line: smoothed data (backward)
    # The plot title shows the estimated age counts from both directions
    p <- ggplot(df, aes(x = idx)) +
        geom_line(aes(y = raw), color = "grey", alpha = 0.4) +
        geom_line(aes(y = fwd), color = "blue", size = 1, alpha = 0.8) +
        geom_line(aes(y = bkwd), color = "red", size = 1, alpha = 0.8) +
        ggtitle(sprintf("fwd: %d, bkwd: %d", fwd_count, bkwd_count)) +
        theme_minimal()
    
    # Display the plot
    print(p)
    # Store the plot in a list for later use
    plts[[i]] <- p
}

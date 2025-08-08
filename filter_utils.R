# ==============================================================================
# filter_utils.R
# ==============================================================================
# About
# -----
# This module provides utilities for signal processing and model selection using
# Fourier basis functions. It includes functions and S3 classes for constructing
# Fourier-based linear models, performing stepwise model selection (forward or
# backward) using AIC or BIC criteria, smoothing 1D data series, and detecting
# peaks in signals. The primary use case is for applications such as automated
# age estimation in biological signals (e.g., shark age estimation from banded
# structures), but the tools are general-purpose for any 1D signal analysis.
# License: MIT License
# Copyright (c) 2025 Angus Lewis

#' Find Indices of Local Maxima (Peaks) in a Numeric Vector
#'
#' This function identifies the indices of local maxima (peaks) in a numeric vector \code{x}.
#' A local maximum is defined as a value that is greater than its immediate neighbors.
#'
#' @param x A numeric vector in which to find local maxima.
#' @param smooth An optional smoothing parameter (currently unused).
#'
#' @return An integer vector containing the indices of local maxima in \code{x}.
#'
#' @examples
#' x <- c(1, 3, 2, 5, 4, 6, 5)
#' find_peaks(x)
#' # Returns: 2 4 6
#'
#' @export
find_peaks <- function(x, smooth = 0) {
  locations <- integer(0)
  for (i in 2:(length(x) - 1)) {
    if (x[i - 1] < x[i] && x[i] > x[i + 1]) {
      locations <- c(locations, i)
    }
  }
  locations
}


#' Check Arguments for Smoothing Function
#'
#' Validates the input arguments for a smoothing function to ensure they meet expected types and constraints.
#'
#' @param series Numeric vector. The data series to be smoothed.
#' @param bandwidth Non-negative integer. The bandwidth parameter for smoothing.
#' @param mode Character string. The smoothing mode, either "forward" or "backward".
#' @param threshold Positive numeric value. The threshold parameter for smoothing.
#' @param criterion Character string. The model selection criterion, either "aic" or "bic".
#'
#' @return None. Throws an error if any argument is invalid.
#' @keywords internal
.check_smooth_arguments <- function(series, bandwidth, mode, threshold, criterion) {
  if (!is.numeric(series) || is.matrix(series) || is.data.frame(series)) {
    stop("Expected series to be a numeric vector")
  }
  if (bandwidth < 0 || bandwidth != as.integer(bandwidth)) {
    stop("Expected bandwidth to be a non-negative integer")
  }
  if (!(mode %in% c("forward", "backward"))) {
    stop('Expected mode to be "forward" or "backward"')
  }
  if (threshold <= 0) {
    stop("Expected threshold to be positive")
  }
  if (!(criterion %in% c("aic", "bic"))) {
    stop('Expected criterion to be "aic" or "bic"')
  }
}

# Constructor function for DFTStats
DFTStats <- function(s2, N_coeffs, N_data) {
  #' Create a DFTStats object
  #' 
  #' DFTStats holds statistics from fitting a linear model, such as variance, 
  #' parameter count, and data size. Provides AIC/BIC model selection criteria.
  #' 
  #' @param s2 Estimated variance of residuals
  #' @param N_coeffs Number of coefficients in the model
  #' @param N_data Number of data points used in the model
  #' @return A DFTStats object
  
  # Create the object as a list with class attribute
  obj <- list(
    s2 = s2,
    N_coeffs = N_coeffs,
    N_data = N_data
  )
  
  # Set the class
  class(obj) <- "DFTStats"
  
  return(obj)
}

# Method to count parameters
count_params <- function(obj) {
  #' Count the number of parameters in the model
  #' 
  #' @param obj A DFTStats object
  #' @return Number of parameters
  UseMethod("count_params")
}

count_params.DFTStats <- function(obj) {
  return(obj$N_coeffs)
}

# Method to compute AIC
aic <- function(obj) {
  #' Compute the Akaike Information Criterion (AIC) for the model
  #' 
  #' @param obj A DFTStats object
  #' @return AIC value
  UseMethod("aic")
}

aic.DFTStats <- function(obj) {
  k <- count_params(obj)
  return(2 * k + obj$N_data * log(obj$s2))
}

# Method to compute BIC
bic <- function(obj) {
  #' Compute the Bayesian Information Criterion (BIC) for the model
  #' 
  #' @param obj A DFTStats object
  #' @return BIC value
  UseMethod("bic")
}

bic.DFTStats <- function(obj) {
  k <- count_params(obj)
  return(k * log(obj$N_data) + obj$N_data * log(obj$s2))
}

# Method to get criterion by name
criterion <- function(obj, which) {
  #' Return the requested model selection criterion
  #' 
  #' @param obj A DFTStats object
  #' @param which Either "aic" or "bic"
  #' @return The requested criterion value
  UseMethod("criterion")
}

criterion.DFTStats <- function(obj, which) {
  if (which == "aic") {
    return(aic(obj))
  } else if (which == "bic") {
    return(bic(obj))
  } else {
    stop(paste("Expected which argument to be either 'aic' or 'bic', got", which))
  }
}

# Print method for nice display
print.DFTStats <- function(x, ...) {
  cat("DFTStats object:\n")
  cat("  Residual variance (s2):", x$s2, "\n")
  cat("  Number of coefficients:", x$N_coeffs, "\n")
  cat("  Number of data points:", x$N_data, "\n")
}

#' DFTOperator Constructor
#'
#' Creates a DFTOperator object for projecting signals onto selected Fourier basis functions using FFT.
#' This manages efficient computation of the Discrete Fourier Transform (DFT) and its inverse for real-valued
#' signals, allowing projection onto a user-specified set of Fourier basis functions.
#'
#' @param N Integer. Length of the signal (number of samples).
#' @param bw Integer. Bandwidth, i.e., the maximum frequency index (number of Fourier basis functions).
#'
#' @return A DFTOperator object (S3 class) containing:
#'   \item{N}{Signal length}
#'   \item{bandwidth}{Number of Fourier basis functions}
#'   \item{x}{Buffer for the input signal}
#'   \item{X}{Buffer for the DFT of the signal}
#'   \item{X_cached}{Cached DFT of the original signal}
#'   \item{xhat}{Buffer for the reconstructed (projected) signal}
#'   \item{projection_idx}{Boolean mask for selected Fourier basis functions}
#'   \item{residuals_calc_malloc}{Buffer for intermediate calculations}
#'   \item{x_sumofsquares}{Sum of squares of the original signal}
#'
#' @details
#' The DFTOperator uses R's built-in FFT functions for efficient computation. The projection is performed
#' by zeroing out unused Fourier coefficients and reconstructing the signal via inverse FFT.
#'
#' @examples
#' op <- DFTOperator(N = 100, bw = 10)
#'
#' @export
DFTOperator <- function(N, bw) {
  # Initialize buffers - R uses complex numbers for FFT
  x <- numeric(N)
  X <- fft(x)  # This will be updated when signal is set
  X_cached <- X
  xhat <- numeric(N)
  
  obj <- list(
    N = N,
    bandwidth = bw,
    x = x,
    X = X,
    X_cached = X_cached,
    xhat = xhat,
    projection_idx = logical(N),
    basis_idx = logical(bw + 1), # +1 for DC component
    residuals_calc_malloc = numeric(N),
    x_sumofsquares = 0.0
  )
  
  class(obj) <- "DFTOperator"
  return(obj)
}

#' Set Fourier Basis Functions for DFTOperator
#'
#' Sets which Fourier basis functions to include in the projection.
#'
#' @param object A DFTOperator object.
#' @param basis_idx Logical vector of length (bandwidth + 1) indicating which Fourier basis 
#'   functions to include (TRUE = include). If NULL, all basis functions up to the bandwidth are included.
#'
#' @return The updated DFTOperator object with modified projection_idx.
#'
#' @details
#' The projection_idx array is used to mask the DFT coefficients for projection.
#' Only the selected basis functions (frequencies) are included in the reconstructed signal.
#'
#' @export
set_basis.DFTOperator <- function(object, basis_idx = NULL) {
  if (is.null(basis_idx)) {
    basis_idx <- rep(TRUE, object$bandwidth + 1)
  }
  
  if (length(basis_idx) != object$bandwidth + 1) {
    stop(sprintf("length of basis must be %d, got %d", object$bandwidth + 1, length(basis_idx)))
  }
  
  # Construct the full projection index for the DFT matrix
  if (length(basis_idx) > 0) {
    object$projection_idx[seq_along(basis_idx)] <- basis_idx
    if (length(basis_idx) > 1) {
      neg_freq_start <- object$N - length(basis_idx) + 2
      neg_freq_end <- object$N
      object$projection_idx[neg_freq_start:neg_freq_end] <- rev(basis_idx[2:length(basis_idx)])
    }
  }
  
  object$basis_idx <- basis_idx
  
  return(object)
}

#' Compute Sum of Squares of Fourier Coefficients
#'
#' Computes the sum of squares of the Fourier coefficients, optionally with masking.
#'
#' @param object A DFTOperator object.
#' @param idx Optional logical vector to select which coefficients to include.
#'
#' @return Numeric value representing the sum of squares of the coefficients.
#'
#' @export
sum_of_squares.DFTOperator <- function(object, idx = NULL) {
  if (is.null(idx)) {
    idx <- rep(TRUE, object$N)
  }
  if (length(idx) != length(object$residuals_calc_malloc)) {
    stop(sprintf("Expected idx to have length %d, got %d", 
                  length(object$residuals_calc_malloc), length(idx)))
  }
  if (length(object$residuals_calc_malloc) > 0) {
    object$residuals_calc_malloc[idx] <- Mod(object$X[idx])^2
    object$residuals_calc_malloc[!idx] <- 0
    s <- sum(object$residuals_calc_malloc)
  } else {
    s <- 0
  }

  return(s)
}

#' Set Signal for DFTOperator
#'
#' Sets the signal for which the projection will be computed and computes its DFT.
#'
#' @param object A DFTOperator object.
#' @param y Numeric vector. The input signal to be projected (1D array of length N).
#'
#' @return The updated DFTOperator object with signal set and DFT computed.
#'
#' @details
#' This function overwrites the internal signal array with the values from y,
#' computes and updates the DFT of the signal, and updates the sum of squares
#' for variance estimation.
#'
#' @export
set_signal.DFTOperator <- function(object, y) {
  if (length(y) != object$N) {
    stop(sprintf("Expected signal length %d, got %d", object$N, length(y)))
  }
  
  object$x <- y
  # Compute the DFT of the signal
  object$X <- fft(object$x)
  object$X_cached <- object$X
  
  # Compute the sum of squares of the signal for variance estimation
  object$x_sumofsquares <- sum_of_squares.DFTOperator(object)
  
  return(object)
}

#' Calculate Residual Sum of Squares for DFTOperator
#'
#' Calculates and returns the residual sum of squares (RSS) for the current projection.
#'
#' @param object A DFTOperator object.
#'
#' @return Numeric value representing the residual sum of squares.
#'
#' @details
#' The RSS is computed as the difference between the total sum of squares and the sum of squares
#' for the projected data, normalized by the number of samples.
#'
#' @export
rss.DFTOperator <- function(object) {
  projected_sos <- sum_of_squares.DFTOperator(object, object$projection_idx)
  rss <- (object$x_sumofsquares - projected_sos) / object$N
  return(rss)
}

#' Project Signal onto Selected Fourier Basis
#'
#' Projects the current signal onto the selected Fourier basis and returns model fit statistics.
#'
#' @param object A DFTOperator object.
#' @param basis_idx Logical vector indicating which Fourier basis functions to include.
#'
#' @return A DFTStats object containing residual variance, number of coefficients, and data size.
#'
#' @details
#' The projection is performed by zeroing out unused Fourier coefficients and reconstructing
#' the signal via inverse FFT. The number of model parameters is twice the number of selected
#' basis functions minus one (for real-valued signals).
#'
#' @export
project.DFTOperator <- function(object, basis_idx) {
  object <- set_basis.DFTOperator(object, basis_idx)

  # Zero out unused coefficients
  X_projected <- object$X_cached
  X_projected[!object$projection_idx] <- 0
  
  # Project the signal onto the selected basis functions
  object$xhat <- Re(fft(X_projected, inverse = TRUE)) / object$N
  
  # Calculate residual variance
  s2 <- rss.DFTOperator(object) / object$N
  
  N_coeffs <- 2 * sum(basis_idx) - 1
  
  return(list(object = object, dft_stats = DFTStats(s2, N_coeffs, object$N)))
}

#' Print method for DFTOperator
#'
#' @param x A DFTOperator object
#' @param ... Additional arguments (ignored)
#' @export
print.DFTOperator <- function(x, ...) {
  cat("DFTOperator object:\n")
  cat("  Signal length (N):", x$N, "\n")
  cat("  Bandwidth:", x$bandwidth, "\n")
  cat("  Sum of squares:", x$x_sumofsquares, "\n")
  if (any(x$projection_idx)) {
    cat("  Active basis functions:", sum(x$projection_idx), "of", length(x$projection_idx), "\n")
  } else {
    cat("  No basis functions selected\n")
  }
}


#' FourierModelBuilder
#'
#' Constructs a FourierModelBuilder object for building and selecting Fourier-based models.
#'
#' @param N Integer. The length of the time series or signal. Must be non-negative.
#' @param max_model_size Integer. The maximum number of Fourier basis functions to use. Must be non-negative and at most N/2.
#' @param mode Character. Specifies the mode or type of model to build (user-defined).
#' @param delta_criteria_threshold Numeric, optional. Threshold for model selection criteria difference (e.g., AIC/BIC). Must be non-negative. Default is 2.
#' @param aic_or_bic Character, optional. Specifies whether to use "aic" or "bic" for model selection. Default is "aic".
#'
#' @return A list of class \code{"FourierModelBuilder"} containing the configuration and state for model building.
#'
#' @details
#' This function initializes a FourierModelBuilder object, which is used to construct and select models based on Fourier projections.
#' It checks input validity and prepares the necessary projection operator and configuration for subsequent modeling steps.
#'
#' @examples
#' builder <- FourierModelBuilder(N = 100, max_model_size = 10, mode = "default")
#'
#' @export
FourierModelBuilder <- function(N, max_model_size, mode, delta_criteria_threshold = 2, aic_or_bic = "aic") {
  if (N < 0) stop("N must be non-negative")
  if (max_model_size > N / 2) stop(sprintf("max_model_size must be at most N/2=%f", N / 2))
  if (max_model_size < 0) stop("max_model_size must be non-negative")
  if (!(aic_or_bic %in% c("aic", "bic"))) stop("aic_or_bic must be either 'aic' or 'bic'")
  if (delta_criteria_threshold < 0) stop("delta_criteria_threshold must be non-negative")
  op <- DFTOperator(N, max_model_size)
  builder <- list(
    N = N,
    max_model_size = max_model_size,
    mode = mode,
    model_size = NULL,
    projection_operator = op,
    dft_stats = NULL,
    aic_or_bic = aic_or_bic,
    delta_criteria_threshold = delta_criteria_threshold,
    basis_idx = NULL,
    y = NULL
  )
  class(builder) <- "FourierModelBuilder"
  builder
}

#' Initialize Projection for FourierModelBuilder
#'
#' This internal function initializes the projection operator and model size for a `FourierModelBuilder` object,
#' depending on the specified mode ("forward" or "backward").
#'
#' @param object A `FourierModelBuilder` object containing at least the following fields:
#'   - `projection_operator`: A Fourier projection operator with a `bandwidth` field.
#'   - `mode`: Character string, either "forward" or "backward".
#'   - `model_size`: (Will be set by this function) The current model size.
#'   - `max_model_size`: The maximum allowed model size (used in "backward" mode).
#'
#' @return The modified `FourierModelBuilder` object with updated `projection_operator` and `model_size`.
#'
#' @details
#' - In "forward" mode, only the first basis function is included and `model_size` is set to 0.
#' - In "backward" mode, all basis functions are included and `model_size` is set to `max_model_size`.
#' - The function updates the `projection_operator` using `set_basis.FourierProjectionOperator`.
#'
#' @keywords internal
#' @seealso \code{\link{set_basis.FourierProjectionOperator}}
#' @examples
#' # Not intended for direct use. See higher-level model builder functions.
.init_proj.FourierModelBuilder <- function(object) {
  bw <- object$projection_operator$bandwidth
  if (object$mode == "forward") {
    object$basis_idx <- rep(FALSE, bw + 1)
    object$basis_idx[1] <- TRUE
    object$model_size <- 0
  } else if (object$mode == "backward") {
    object$basis_idx <- rep(TRUE, bw + 1)
    object$model_size <- object$max_model_size
  } else {
    stop("mode can be 'forward' or 'backward' only")
  }
  object$projection_operator <- set_basis.DFTOperator(object$projection_operator, object$basis_idx)
  object
}

#' Reset a FourierModelBuilder Object
#'
#' This function resets a `FourierModelBuilder` object by reinitializing its projection
#' and clearing any existing dft stats.
#'
#' @param object A `FourierModelBuilder` object to be reset.
#'
#' @return The reset `FourierModelBuilder` object with its projection reinitialized and
#'         the `dft_stats` field set to `NULL`.
#'
#' @examples
#' # Assuming `fmb` is a FourierModelBuilder object:
#' fmb <- reset.FourierModelBuilder(fmb)
#'
#' @export
reset.FourierModelBuilder <- function(object) {
  object <- .init_proj.FourierModelBuilder(object)
  object$dft_stats <- NULL
  object
}

#' Initialise a FourierModelBuilder object
#'
#' This function initialises a \code{FourierModelBuilder} object by assigning the response variable \code{y},
#' resetting the object to its initial state, and computing the initial dft stats using the Fourier projection operator.
#'
#' @param object A \code{FourierModelBuilder} object to be initialised.
#' @param y A numeric vector representing the response variable.
#'
#' @return The initialised \code{FourierModelBuilder} object with updated fields.
#'
#' @seealso \code{\link{reset.FourierModelBuilder}}, \code{\link{apply.FourierProjectionOperator}}
#' @export
initialise.FourierModelBuilder <- function(object, y) {
  object$y <- y
  object <- reset.FourierModelBuilder(object)
  object$projection_operator <- set_signal.DFTOperator(object$projection_operator, y)
  res <- project.DFTOperator(object$projection_operator, object$projection_operator$basis_idx)
  object$projection_operator <- res$object
  object$dft_stats <- res$dft_stats
  object
}

#' Perform a single iteration of model selection for a FourierModelBuilder object.
#'
#' This function attempts to improve the current linear model by either adding or removing a basis function,
#' depending on the specified mode ("forward" or "backward"). It evaluates all possible single-basis changes,
#' selects the one that optimizes the specified criterion (AIC or BIC), and updates the model if the improvement
#' exceeds a given threshold.
#'
#' @param object A \code{FourierModelBuilder} object containing the current model, projection operator, mode,
#'        criterion type, and threshold for model update.
#'
#' @return A list with two elements:
#'   \item{object}{The updated \code{FourierModelBuilder} object (if an update occurred).}
#'   \item{updated}{Logical; \code{TRUE} if the model was updated, \code{FALSE} otherwise.}
#'
#' @details
#' In "forward" mode, the function attempts to add a basis function to the model if it improves the criterion
#' by more than \code{delta_criteria_threshold}. In "backward" mode, it attempts to remove a basis function if
#' the criterion does not worsen by more than \code{delta_criteria_threshold}. The function uses helper methods
#' such as \code{set_basis.FourierProjectionOperator}, \code{apply.FourierProjectionOperator}, and
#' \code{criterion.LinearModel} to manipulate the projection operator and evaluate model quality.
#'
#' @seealso \code{\link{set_basis.FourierProjectionOperator}}, \code{\link{apply.FourierProjectionOperator}}, \code{\link{criterion.LinearModel}}
#'
#' @examples
#' # Not run:
#' # result <- iteration_.FourierModelBuilder(my_fourier_model_builder)
#'
#' @export
iteration_.FourierModelBuilder <- function(object) {
  new_basis <- object$projection_operator$basis_idx
  best_stats <- NULL
  best_criterion <- Inf
  best_idx <- NULL
  for (idx in seq_along(new_basis)) {
    is_active <- new_basis[idx]
    if (idx == 1 ||
        (object$mode == "forward" && is_active) ||
        (object$mode == "backward" && !is_active)) {
      next
    }
    new_basis[idx] <- !new_basis[idx]
    temp_op <- set_basis.DFTOperator(object$projection_operator, new_basis)
    temp_op <- set_signal.DFTOperator(temp_op, object$y)
    res <- project.DFTOperator(temp_op, new_basis)
    new_stats <- res$dft_stats
    crit <- criterion.DFTStats(new_stats, object$aic_or_bic)
    if (crit < best_criterion) {
      best_criterion <- crit
      best_stats <- new_stats
      best_idx <- idx
    }
    new_basis[idx] <- !new_basis[idx]
  }
  delta_criteria <- criterion.DFTStats(object$dft_stats, object$aic_or_bic) - best_criterion
  if (object$mode == "forward" && (delta_criteria > object$delta_criteria_threshold)) {
    object$dft_stats <- best_stats
    object$model_size <- object$model_size + 1
    new_basis[best_idx] <- TRUE
    object$basis_idx <- new_basis
    object$projection_operator <- set_basis.DFTOperator(object$projection_operator, new_basis)
    return(list(object = object, updated = TRUE))
  } else if (object$mode == "backward" && (delta_criteria > -object$delta_criteria_threshold)) {
    object$dft_stats <- best_stats
    object$model_size <- object$model_size - 1
    new_basis[best_idx] <- FALSE
    object$basis_idx <- new_basis
    object$projection_operator <- set_basis.DFTOperator(object$projection_operator, new_basis)
    return(list(object = object, updated = TRUE))
  }
  list(object = object, updated = FALSE)
}

#' Perform a single iteration step for the FourierModelBuilder object.
#'
#' This function manages the iteration process for a FourierModelBuilder object,
#' handling both "forward" and "backward" modes. In "forward" mode, it adds terms
#' to the model if the current model size is less than the maximum allowed. In
#' "backward" mode, it removes terms if the model size is greater than zero.
#'
#' @param object A FourierModelBuilder object containing the current model state,
#'   including the linear model, mode ("forward" or "backward"), model size, and
#'   maximum model size.
#'
#' @return A list with two elements:
#'   \describe{
#'     \item{object}{The updated FourierModelBuilder object.}
#'     \item{updated}{Logical indicating whether the model was updated in this iteration.}
#'   }
#'
#' @details
#' The function checks if the linear model is initialized. Depending on the mode,
#' it calls \code{iteration_.FourierModelBuilder()} to perform the actual update.
#' If no update is possible, it returns the object unchanged with \code{updated = FALSE}.
#'
#' @seealso \code{\link{iteration_.FourierModelBuilder}}
#'
#' @examples
#' # Assuming 'builder' is a properly initialized FourierModelBuilder object:
#' result <- iteration.FourierModelBuilder(builder)
#' updated_builder <- result$object
#' was_updated <- result$updated
iteration.FourierModelBuilder <- function(object) {
  if (is.null(object$dft_stats)) stop("Expected dft_stats to be initialised")
  if (object$mode == "forward" && object$model_size < object$max_model_size) {
    res <- iteration_.FourierModelBuilder(object)
    object <- res$object
    return(list(object = object, updated = res$updated))
  } else if (object$mode == "backward" && object$model_size > 0) {
    res <- iteration_.FourierModelBuilder(object)
    object <- res$object
    return(list(object = object, updated = res$updated))
  }
  list(object = object, updated = FALSE)
}

#' Build a Fourier Model using Iterative Updates
#'
#' This function constructs a Fourier model by initializing the model builder object
#' and then iteratively updating it until convergence. The process involves repeatedly
#' calling an iteration function that updates the model builder object and checks if
#' further updates are necessary. Once the model is fully built, the resulting linear
#' model is returned.
#'
#' @param object A list or object containing the necessary components for building the Fourier model,
#'   including the response variable \code{y} and any other required parameters.
#'
#' @return The final linear model object resulting from the iterative Fourier model building process.
#'
#' @details
#' The function first initializes the model builder using \code{initialise.FourierModelBuilder}.
#' It then enters a loop, where \code{iteration.FourierModelBuilder} is called to update the model.
#' The loop continues until no further updates are made (i.e., \code{res$updated} is \code{FALSE}).
#'
#' @seealso \code{\link{initialise.FourierModelBuilder}}, \code{\link{iteration.FourierModelBuilder}}
#'
#' @examples
#' # Assuming 'builder' is a properly constructed FourierModelBuilder object:
#' # model <- build.FourierModelBuilder(builder)
build.FourierModelBuilder <- function(object) {
  object <- initialise.FourierModelBuilder(object, object$y)
  repeat {
    res <- iteration.FourierModelBuilder(object)
    object <- res$object
    if (!res$updated) break
  }
  object
}

#' Smooth a 1D Data Series Using Stepwise Selection of Fourier Basis Functions
#'
#' Applies a stepwise selection algorithm to fit a Fourier basis model to a 1D data series, 
#' allowing for flexible smoothing based on information criteria.
#'
#' @param series Numeric vector. The 1D data series to be smoothed.
#' @param bandwidth Integer. The maximum number of Fourier basis functions to consider.
#' @param mode Character. The stepwise selection mode, either "forward" or "backward". Default is "forward".
#' @param threshold Numeric. The threshold for including/excluding basis functions during selection. Default is 2.0.
#' @param criterion Character. The information criterion to use for model selection, e.g., "aic" or "bic". Default is "aic".
#'
#' @return An object of class \code{FourierModelBuilder} containing the fitted model and selection details.
#'
#' @details
#' This function constructs a Fourier basis model for the input series using stepwise selection
#' (forward or backward) based on the specified information criterion. The process iteratively adds or removes
#' basis functions to optimize model fit while controlling for overfitting.
#'
#' @seealso \code{\link{FourierModelBuilder}}, \code{\link{initialise.FourierModelBuilder}}, \code{\link{build.FourierModelBuilder}}
#'
#' @examples
#' series <- sin(seq(0, 2 * pi, length.out = 100)) + rnorm(100, 0, 0.1)
#' model <- smooth(series, bandwidth = 10, mode = "forward", threshold = 2.0, criterion = "aic")
#'
#' @export
#' Smooth a 1D data series using stepwise selection of Fourier basis functions
smooth <- function(series, bandwidth, mode = "forward", threshold = 2.0, criterion = "aic") {
  .check_smooth_arguments(series, bandwidth, mode, threshold, criterion)
  N <- length(series)
  model_builder <- FourierModelBuilder(N, bandwidth, mode, threshold, criterion)
  model_builder$y <- series
  model_builder <- build.FourierModelBuilder(model_builder)
  dft_stats <- model_builder$dft_stats
  
  # Return both the fitted signal and the statistics
  # Update the final projection to get the fitted signal
  model_builder$projection_operator <- set_signal.DFTOperator(model_builder$projection_operator, series)
  res <- project.DFTOperator(model_builder$projection_operator, model_builder$basis_idx)
  model_builder$projection_operator <- res$object
  fitted_signal <- model_builder$projection_operator$xhat
  
  list(fitted = fitted_signal, dft_stats = dft_stats)
}

#' Estimate Age by Counting Peaks in a Smoothed Signal
#'
#' This function estimates the "age" of a shark by counting the number of peaks in a smoothed version of the input signal.
#' The smoothing is performed using the specified parameters, and peaks are identified in the fitted (smoothed) signal.
#'
#' @param series Numeric vector. The input signal (e.g., growth band data) to be analyzed.
#' @param max_age Integer. The maximum age (or number of peaks) to consider during smoothing.
#' @param mode Character. The smoothing mode to use. Default is \code{"forward"}.
#' @param threshold Numeric. The threshold value for peak detection. Default is \code{2.0}.
#' @param criterion Character. The criterion for model selection during smoothing (e.g., \code{"aic"}). Default is \code{"aic"}.
#'
#' @return A list with the following elements:
#'   \item{age}{Integer. The estimated age (number of peaks detected).}
#'   \item{peak_indices}{Integer vector. The indices of the detected peaks in the fitted signal.}
#'   \item{fitted}{Numeric vector. The smoothed (fitted) signal.}
#'
#' @details
#' The function first smooths the input signal using the \code{smooth} function with the provided parameters.
#' It then detects peaks in the smoothed signal using the \code{find_peaks} function.
#' The number of detected peaks is returned as the estimated age.
#'
#' @seealso \code{\link{smooth}}, \code{\link{find_peaks}}
#'
#' @examples
#' series <- c(1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3,
#'    1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3)
#' result <- age_shark(series, max_age = 5)
#' print(result$age)
#' print(result$peak_indices)
#' plot(series, type = "l")
#' lines(result$fitted, col = "blue")
#' Estimate "age" by counting peaks in a smoothed signal
age_shark <- function(series, max_age, mode = "forward", threshold = 2.0, criterion = "aic") {
  # Smooth the input series using stepwise Fourier basis selection
  smooth_result <- smooth(series, max_age, mode, threshold, criterion)
  # Find peaks in the smoothed (fitted) signal
  peaks <- find_peaks(smooth_result$fitted)
  # Return the number of peaks, their indices, and the fitted signal
  list(age = length(peaks), peak_indices = peaks, fitted = smooth_result$fitted)
}
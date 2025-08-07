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

#' Check if a vector is of complex type
#'
#' This function checks whether the input vector \code{a} is of complex type.
#'
#' @param a A vector to be checked.
#'
#' @return A logical value: \code{TRUE} if \code{a} is of complex type, \code{FALSE} otherwise.
#'
#' @examples
#' is_complex_eltype(1+2i)    # TRUE
#' is_complex_eltype(1:5)     # FALSE
#' is_complex_eltype(c(1, 2)) # FALSE
#'
#' @export
#' Check if a vector is complex
is_complex_eltype <- function(a) {
  is.complex(a)
}

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

#' Create a LinearModel object
#'
#' Constructs a list representing a linear model with coefficients, fitted values, observed values, residuals, and residual variance.
#'
#' @param coeffs Numeric vector of model coefficients.
#' @param fitted Numeric vector of fitted values from the model.
#' @param y Numeric vector of observed response values.
#' @param residuals Numeric vector of residuals (observed - fitted).
#' @param s2 Numeric value representing the residual variance.
#'
#' @return An object of class \code{LinearModel}, which is a list containing:
#'   \item{coeffs}{Model coefficients}
#'   \item{fitted}{Fitted values}
#'   \item{y}{Observed response values}
#'   \item{residuals}{Residuals}
#'   \item{s2}{Residual variance}
#'   \item{.ll}{Log-likelihood (initialized as NULL)}
#'   \item{.aic}{Akaike Information Criterion (initialized as NULL)}
#'   \item{.bic}{Bayesian Information Criterion (initialized as NULL)}
#'
#' @examples
#' coeffs <- c(Intercept = 1.5, Slope = 0.8)
#' fitted <- c(2.3, 3.1, 4.0)
#' y <- c(2.5, 3.0, 4.2)
#' residuals <- y - fitted
#' s2 <- var(residuals)
#' model <- LinearModel(coeffs, fitted, y, residuals, s2)
#'
#' @export
LinearModel <- function(coeffs, fitted, y, residuals, s2) {
  structure(list(
    coeffs = coeffs,
    fitted = fitted,
    y = y,
    residuals = residuals,
    s2 = s2,
    .ll = NULL,
    .aic = NULL,
    .bic = NULL
  ), class = "LinearModel")
}

#' Count the Number of Parameters in a LinearModel Object
#'
#' This function calculates the number of parameters in a `LinearModel` object.
#' If the coefficients are of a complex element type, each coefficient is counted as two parameters (real and imaginary parts).
#'
#' @param object A `LinearModel` object containing a `coeffs` element.
#'
#' @return An integer representing the number of parameters in the model.
#'
#' @examples
#' # Assuming `lm_obj` is a LinearModel object with real coefficients:
#' count_params.LinearModel(lm_obj)
#'
#' # If `lm_obj` has complex coefficients:
#' count_params.LinearModel(lm_obj)
#'
#' @seealso \code{\link{is_complex_eltype}}
#' @export
count_params.LinearModel <- function(object) {
  if (is_complex_eltype(object$coeffs)) {
    k <- 2 * length(object$coeffs)
  } else {
    k <- length(object$coeffs)
  }
  k
}

#' Compute the Log-Likelihood for a Linear Model Object
#'
#' Calculates and caches the log-likelihood of a fitted linear model object,
#' assuming normally distributed errors. If the log-likelihood has already been
#' computed and stored in the object (`.ll`), it is returned directly.
#'
#' @param object A linear model object containing at least the components:
#'   \describe{
#'     \item{residuals}{A numeric vector of model residuals.}
#'     \item{s2}{The estimated variance of the residuals.}
#'     \item{.ll}{(Optional) Cached log-likelihood value.}
#'   }
#'
#' @return The log-likelihood (numeric scalar) of the model under the normal error assumption.
#' @examples
#' # Assuming 'fit' is a linear model object with required components:
#' # loglikelihood.LinearModel(fit)
#' @export
loglikelihood.LinearModel <- function(object) {
  if (is.null(object$.ll)) {
    # Use the sum of log pdfs for the residuals (normal errors)
    object$.ll <- sum(dnorm(object$residuals, sd = sqrt(object$s2), log = TRUE))
  }
  object$.ll
}

#' Calculate the Akaike Information Criterion (AIC) for a LinearModel object
#'
#' Computes the AIC value for a given LinearModel object. If the AIC has already been calculated and stored in the object, it returns the cached value. Otherwise, it calculates the number of parameters using \code{count_params.LinearModel}, computes the log-likelihood using \code{loglikelihood.LinearModel}, and then calculates the AIC as \code{2 * k - 2 * loglikelihood}.
#'
#' @param object A \code{LinearModel} object for which the AIC is to be calculated.
#' @return Numeric value representing the AIC of the model.
#' @examples
#' # Assuming 'model' is a LinearModel object
#' aic_value <- aic.LinearModel(model)
#' @export
aic.LinearModel <- function(object) {
  if (is.null(object$.aic)) {
    k <- count_params.LinearModel(object)
    object$.aic <- 2 * k - 2 * loglikelihood.LinearModel(object)
  }
  object$.aic
}

#' Compute the Bayesian Information Criterion (BIC) for a LinearModel object
#'
#' This function calculates the BIC for a given \code{LinearModel} object. If the BIC has already been computed and stored in the object, it returns the cached value. Otherwise, it computes the BIC using the number of parameters, the sample size, and the log-likelihood of the model.
#'
#' @param object A \code{LinearModel} object for which the BIC is to be calculated.
#'
#' @return The Bayesian Information Criterion (BIC) value for the model.
#'
#' @details
#' The BIC is calculated as \eqn{k \log(n) - 2 \log L}, where \eqn{k} is the number of parameters in the model, \eqn{n} is the number of observations, and \eqn{L} is the likelihood of the model.
#'
#' @seealso \code{\link{count_params.LinearModel}}, \code{\link{loglikelihood.LinearModel}}
#'
#' @examples
#' # Assuming 'model' is a LinearModel object:
#' # bic.LinearModel(model)
bic.LinearModel <- function(object) {
  if (is.null(object$.bic)) {
    k <- count_params.LinearModel(object)
    n <- length(object$y)
    object$.bic <- k * log(n) - 2 * loglikelihood.LinearModel(object)
  }
  object$.bic
}

#' Compute Model Selection Criterion for LinearModel Objects
#'
#' This function computes a specified model selection criterion (AIC or BIC) for a given `LinearModel` object.
#'
#' @param object A `LinearModel` object for which the criterion is to be computed.
#' @param which A character string specifying the criterion to compute. Must be either `"aic"` or `"bic"`.
#'
#' @return The computed value of the specified criterion (AIC or BIC) for the provided model.
#'
#' @details
#' The function dispatches to either `aic.LinearModel` or `bic.LinearModel` based on the value of `which`.
#' If `which` is not `"aic"` or `"bic"`, the function will stop with an error.
#'
#' @examples
#' # Assuming `fit` is a LinearModel object:
#' # criterion.LinearModel(fit, "aic")
#' # criterion.LinearModel(fit, "bic")
#'
#' @export
criterion.LinearModel <- function(object, which) {
  if (which == "aic") {
    aic.LinearModel(object)
  } else if (which == "bic") {
    bic.LinearModel(object)
  } else {
    stop('Expected which argument to be either "aic" or "bic"')
  }
}

#' Compute the Discrete Fourier Transform (DFT) matrix of size N x N
#'
#' This function generates the DFT matrix, which is commonly used in signal processing
#' and Fourier analysis. The resulting matrix can be used to compute the DFT of a vector
#' via matrix multiplication.
#'
#' @param N Integer. The size of the DFT matrix (number of rows and columns).
#'
#' @return A complex matrix of dimensions N x N representing the DFT matrix.
#'
#' @examples
#' dft_matrix(4)
#'
#' @export
dft_matrix <- function(N) {
  n <- 0:(N-1)
  k <- 0:(N-1)
  W <- outer(n, k, function(n, k) exp(-2i * pi * n * k / N))
  return(W)
}

#' Create a Fourier Projection Operator
#'
#' Constructs a FourierProjectionOperator object for projecting signals onto a
#' Fourier basis with a specified bandwidth.
#'
#' @param N Integer. The length of the signal or the number of points in the basis.
#' @param bw Integer. The bandwidth, i.e., the number of Fourier basis functions to use.
#'
#' @return A list of class \code{"FourierProjectionOperator"} containing:
#'   \item{N}{The length of the signal.}
#'   \item{bandwidth}{The bandwidth (number of basis functions).}
#'   \item{dft_mat}{The discrete Fourier transform matrix of size \code{N}.}
#'   \item{design}{Reserved for design matrix (default \code{NULL}).}
#'   \item{coef_operator}{Reserved for coefficient operator (default \code{NULL}).}
#'   \item{basis_idx}{Reserved for basis indices (default \code{NULL}).}
#'   \item{projection_operator}{Reserved for projection operator (default \code{NULL}).}
#'
#' @examples
#' op <- FourierProjectionOperator(N = 128, bw = 10)
#'
#' @export
FourierProjectionOperator <- function(N, bw) {
  dft_mat <- dft_matrix(N)
  op <- list(
    N = N,
    bandwidth = bw,
    dft_mat = dft_mat,
    design = NULL,
    coef_operator = NULL,
    basis_idx = NULL,
    projection_operator = NULL
  )
  class(op) <- "FourierProjectionOperator"
  op
}

#' Set the Basis for a FourierProjectionOperator Object
#'
#' This function updates a `FourierProjectionOperator` object by selecting a subset of basis functions
#' (Fourier coefficients) to use in the projection. It constructs the corresponding design matrix,
#' coefficient operator, and projection operator based on the selected basis indices.
#'
#' @param object A `FourierProjectionOperator` object containing the DFT matrix (`dft_mat`), bandwidth, and other relevant fields.
#' @param basis_idx Logical or integer vector indicating which basis functions to include. If `NULL`, all basis functions up to the object's bandwidth are included.
#'
#' @return The modified `FourierProjectionOperator` object with updated fields:
#'   \item{design}{The design matrix corresponding to the selected basis functions.}
#'   \item{coef_operator}{The coefficient operator for projecting onto the selected basis.}
#'   \item{basis_idx}{The indices of the selected basis functions.}
#'   \item{projection_operator}{The projection operator matrix.}
#'
#' @details
#' The function ensures conjugate symmetry by mirroring the selected basis indices, which is important for real-valued signals.
#' It also checks that the number of selected basis functions does not exceed the allowed bandwidth.
#'
#' @examples
#' # Assuming 'op' is a valid FourierProjectionOperator object:
#' op <- set_basis.FourierProjectionOperator(op, basis_idx = c(TRUE, FALSE, TRUE))
#'
#' @export
set_basis.FourierProjectionOperator <- function(object, basis_idx = NULL) {
  if (is.null(basis_idx)) {
    basis_idx <- rep(TRUE, object$bandwidth + 1)
  }
  if (length(basis_idx) > object$bandwidth + 1) {
    stop(sprintf("length of basis must be less than %d.", object$bandwidth + 1))
  }
  # Construct projection index (mirror for conjugate symmetry)
  projection_idx <- c(
    basis_idx,
    rep(0, object$N - object$bandwidth * 2 - 1),
    rev(basis_idx[-1])
  )
  design <- object$dft_mat[as.logical(projection_idx), , drop = FALSE]
  coef_operator <- design / ncol(design)
  projection_operator <- Conj(t(design)) %*% coef_operator
  object$design <- design
  object$coef_operator <- coef_operator
  object$basis_idx <- basis_idx
  object$projection_operator <- projection_operator
  object
}

#' Apply the Fourier Projection Operator to a vector
#'
#' This function applies a precomputed Fourier projection operator to the input vector \code{y}.
#' It projects \code{y} onto the Fourier basis defined in the \code{object}, computes the coefficients,
#' the projection, residuals, and the estimated variance of the residuals.
#'
#' @param object A list containing the Fourier basis and projection operators. Must include
#'   \code{coef_operator} (matrix for computing coefficients) and \code{projection_operator}
#'   (matrix for projecting onto the basis).
#' @param y A numeric or complex vector to be projected.
#'
#' @return A \code{LinearModel} object containing:
#'   \item{coeffs}{The coefficients of the projection in the Fourier basis.}
#'   \item{proj}{The projection of \code{y} onto the Fourier basis.}
#'   \item{y}{The original input vector.}
#'   \item{residuals}{The residuals from the projection.}
#'   \item{s2}{The estimated variance of the residuals.}
#'
#' @details
#' If the coefficients are complex, the degrees of freedom for the variance estimate are adjusted accordingly.
#' The function requires that the basis has been set in \code{object} prior to calling.
#'
#' @seealso \code{\link{set_basis}}, \code{\link{LinearModel}}
#'
#' @examples
#' # Assuming 'obj' is a properly initialized object with basis set
#' y <- rnorm(100)
#' result <- apply.FourierProjectionOperator(obj, y)
apply.FourierProjectionOperator <- function(object, y) {
  if (is.null(object$coef_operator)) {
    stop("Choose basis first (call set_basis)")
  }
  coeffs <- as.vector(object$coef_operator %*% y)
  proj <- Re(object$projection_operator %*% y)
  residuals <- y - proj
  sse <- sum(residuals^2)
  if (is_complex_eltype(coeffs)) {
    s2 <- sse / (length(y) - 2 * length(coeffs))
  } else {
    s2 <- sse / (length(y) - length(coeffs))
  }
  LinearModel(coeffs, proj, y, residuals, s2)
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
  op <- FourierProjectionOperator(N, max_model_size)
  builder <- list(
    N = N,
    max_model_size = max_model_size,
    mode = mode,
    model_size = NULL,
    projection_operator = op,
    linear_model = NULL,
    aic_or_bic = aic_or_bic,
    delta_criteria_threshold = delta_criteria_threshold,
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
    basis_idx <- rep(FALSE, bw + 1)
    basis_idx[1] <- TRUE
    object$model_size <- 0
  } else if (object$mode == "backward") {
    basis_idx <- rep(TRUE, bw + 1)
    object$model_size <- object$max_model_size
  } else {
    stop("mode can be 'forward' or 'backward' only")
  }
  object$projection_operator <- set_basis.FourierProjectionOperator(object$projection_operator, basis_idx)
  object
}

#' Reset a FourierModelBuilder Object
#'
#' This function resets a `FourierModelBuilder` object by reinitializing its projection
#' and clearing any existing linear model.
#'
#' @param object A `FourierModelBuilder` object to be reset.
#'
#' @return The reset `FourierModelBuilder` object with its projection reinitialized and
#'         the `linear_model` field set to `NULL`.
#'
#' @examples
#' # Assuming `fmb` is a FourierModelBuilder object:
#' fmb <- reset.FourierModelBuilder(fmb)
#'
#' @export
reset.FourierModelBuilder <- function(object) {
  object <- .init_proj.FourierModelBuilder(object)
  object$linear_model <- NULL
  object
}

#' Initialise a FourierModelBuilder object
#'
#' This function initialises a \code{FourierModelBuilder} object by assigning the response variable \code{y},
#' resetting the object to its initial state, and computing the initial linear model using the Fourier projection operator.
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
  object$linear_model <- apply.FourierProjectionOperator(object$projection_operator, y)
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
  best_model <- NULL
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
    temp_op <- set_basis.FourierProjectionOperator(object$projection_operator, new_basis)
    new_lm <- apply.FourierProjectionOperator(temp_op, object$y)
    crit <- criterion.LinearModel(new_lm, object$aic_or_bic)
    if (crit < best_criterion) {
      best_criterion <- crit
      best_model <- new_lm
      best_idx <- idx
    }
    new_basis[idx] <- !new_basis[idx]
  }
  delta_criteria <- criterion.LinearModel(object$linear_model, object$aic_or_bic) - best_criterion
  if (object$mode == "forward" && (delta_criteria > object$delta_criteria_threshold)) {
    object$linear_model <- best_model
    object$model_size <- object$model_size + 1
    new_basis[best_idx] <- TRUE
    object$projection_operator <- set_basis.FourierProjectionOperator(object$projection_operator, new_basis)
    return(list(object = object, updated = TRUE))
  } else if (object$mode == "backward" && (delta_criteria > -object$delta_criteria_threshold)) {
    object$linear_model <- best_model
    object$model_size <- object$model_size - 1
    new_basis[best_idx] <- FALSE
    object$projection_operator <- set_basis.FourierProjectionOperator(object$projection_operator, new_basis)
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
  if (is.null(object$linear_model)) stop("Expected linear model to be initialised")
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
  object$linear_model
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
  model_builder <- initialise.FourierModelBuilder(model_builder, series)
  model_builder <- build.FourierModelBuilder(model_builder)
  model_builder
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
  lm <- smooth(series, max_age, mode, threshold, criterion)
  peaks <- find_peaks(lm$fitted)
  list(age = length(peaks), peak_indices = peaks, fitted = lm$fitted)
}
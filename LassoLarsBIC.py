# Implement LassoLarsIC but for a non-standard case of BIC.
# Copyright (C) 2025  Angus Lewis

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from sklearn.linear_model import LassoLars, LassoLarsIC, lars_path
from sklearn.utils.validation import validate_data
from sklearn.linear_model._base import _preprocess_data, _fit_context

def _no_penalty(x):
    return 1

def get_active_set(coef):
    mask = np.abs(coef) > np.finfo(coef.dtype).eps
    return mask

class LassoLarsBIC(LassoLarsIC, LassoLars):
    """This is the same as LassoLarsIC from sklearn, except the BIC is calculated in 
    a different way (to include variance explicitly as a parameter, and the L1 loss as a prior).

    Based on LassoLarsIC. Credit to
    Authors: The scikit-learn developers
    Their licence is
    SPDX-License-Identifier: BSD-3-Clause
    """

    _parameter_constraints: dict = {
        **LassoLarsIC._parameter_constraints,
    }

    def __init__(
        self,
        criterion="bic",
        *,
        fit_intercept=True,
        verbose=False,
        precompute="auto",
        max_iter=500,
        eps=np.finfo(float).eps,
        copy_X=True,
        positive=False,
        noise_variance=None,
    ):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.max_iter = max_iter
        self.verbose = verbose
        self.copy_X = copy_X
        self.precompute = precompute
        self.eps = eps
        self.fit_path = True
        self.noise_variance = noise_variance

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, prior=_no_penalty, copy_X=None):
        """Fit the model using X, y as training data.
        To select the regularisation parameter, alpha, this class uses BIC calculated as follows. 
        If the noise variance is not known (noise_variance=None on construction),
        BIC = (
            N*log(2 * pi * sigma^2)  # likelihood
            + log(N)*dof             # BIC penalty
            + log(prior(beta))       # optional prior on models
        )
        where N is the number of samples, sigma is an estimate of the variance of the residuals, 
        beta are the regression coefficients, dof is the degrees of freedom (which is the dof as in [1], 
        plus 1 for the variance term) and prior(beta) an additional [optional] prior distribution for 
        the betas. This is different to the BIC in [1] as here we treat sigma as a parameter to be inferred.
        
        If the noise variance is known,
        BIC = (
             N*log(2 * pi * sigma^2)  # likelihood
            + RSS / sigma^2.          # likelihood
            + log(N)*dof              # BIC penalty
            + log(prior(beta))        # optional prior on models
        )
        where RSS is the residual sum of squares ||y-X*beta||^2 and sigma is known. This is the same as in [1].


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        prior : callable, default=_no_penalty
            An additional penalty term for BIC; the log of penalty is added to the BIC.
            This can be used to include a prior distribution on the models.
            The default is the constant function _no_prior(x) = 1 (so log(penalty))=0.
            The argument passed to prior is the sequence of model parameters produced by LARS
            and prior should return and ndarray containing the pdf of the prior for each model
            in the sequence.

        copy_X : bool, default=None
            If provided, this parameter will override the choice
            of copy_X made at instance creation.
            If ``True``, X will be copied; else, it may be overwritten.

        Returns
        -------
        self : object
            Returns an instance of self.

        [1] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
            "On the degrees of freedom of the lasso."
            The Annals of Statistics 35.5 (2007): 2173-2192.
            <0712.0881>
        """
        if copy_X is None:
            copy_X = self.copy_X
        X, y = validate_data(self, X, y, force_writeable=True, y_numeric=True)

        X, y, Xmean, ymean, Xstd = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=copy_X
        )

        Gram = self.precompute

        alphas_, _, coef_path_, self.n_iter_ = lars_path(
            X,
            y,
            Gram=Gram,
            copy_X=copy_X,
            copy_Gram=True,
            alpha_min=0.0,
            method="lasso",
            verbose=self.verbose,
            max_iter=self.max_iter,
            eps=self.eps,
            return_n_iter=True,
            positive=self.positive,
        )

        n_samples = X.shape[0]

        if self.criterion == "bic":
            criterion_factor = np.log(n_samples)
        elif self.criterion == "aic":
            raise ValueError(
                f"criterion should be bic, got {self.criterion} (aic not implemented)"
            )
            # criterion_factor = 2
        else:
            raise ValueError(
                f"criterion should be bic, got {self.criterion}"
            )

        preds_ = np.dot(X, coef_path_)
        residuals = y[:, np.newaxis] - preds_
        residuals_sum_squares = np.sum(residuals**2, axis=0)

        # degrees of freedom as in [1]
        degrees_of_freedom = np.zeros(coef_path_.shape[1], dtype=int)
        for k, coef in enumerate(coef_path_.T):
            mask = np.abs(coef) > np.finfo(coef.dtype).eps
            # get the number of degrees of freedom equal to:
            # Xc = X[:, mask]
            # Trace(Xc * inv(Xc.T, Xc) * Xc.T) ie the number of non-zero coefs
            # and include the variance if it is estimated too
            degrees_of_freedom[k] = np.sum(mask) + (self.noise_variance is None)

        self.alphas_ = alphas_

        # compute BIC including l1 lasso loss (a prior on the betas) and 
        # the optional additional prior on the model
        # l1_penalty_ = np.sum(np.abs(coef_path_), axis=0)
        prior_penalty_ = np.log(prior(coef_path_))
        if self.noise_variance is None:
            self.noise_variance_ = residuals_sum_squares / n_samples
            
            self.criterion_ = (
                -2*( # -2 x loglikelihood terms
                    - 0.5*n_samples*np.log(self.noise_variance_)  # likelihood
                    # # The terms below are for the prior, but these doubly penalise the coefficients, so leave them out
                    # + degrees_of_freedom*np.log(alphas_) # prior
                    # # lars_path uses 1/(2*n_samples) RSS + alpha * ||beta||_1, so need to multiply alpha by 2*n_samples
                    # - 2*n_samples*alphas_*l1_penalty_ # L1 penalty/prior on betas
                    + prior_penalty_ # optional prior on models
                )
                + criterion_factor*degrees_of_freedom # BIC penalty
            )
        else:
            self.noise_variance_ = np.full(self.noise_variance, coef_path_.shape[1])
            self.criterion_ = (
                n_samples * np.log(self.noise_variance_) # likelihood
                + residuals_sum_squares / self.noise_variance_ # likelihood
                + criterion_factor * degrees_of_freedom # BIC penalty
                + prior_penalty_ # optional prior on models
            )

        n_best = np.argmin(self.criterion_)

        self.alpha_ = alphas_[n_best]
        self.coef_ = coef_path_[:, n_best]
        self._set_intercept(Xmean, ymean, Xstd)
        self.variance_estimate = self.noise_variance_[n_best]
        return self
import numpy as np
import particles
from particles import smc_samplers as ssp
from particles import distributions as dists
from scipy import stats

def simulate_posterior_fast(alpha, nboot, y, X, sigma2=None, lam=0, mcmc_len = 16, n_smc=10):
    print("Computing Ridge estimate")
    W = np.linalg.inv(np.dot(X.T, X) +(lam*np.eye(X.shape[1])))
    mu = (W @ X.T) @ y

    print("Computing residual variance")
    if sigma2 is None:
        preds = X @ mu
        sigma2 = np.sum((y - preds)**2)/len(y)
    
    print("Computing covariance of params")
    S = sigma2*W*np.dot(X.T, X)*W.T
    
    # Simulate from posterior approximation which is proportional to a Gaussian multiplied by Laplacian prior
    # The rate parameter of lars_path is scaled so undo scaling here
    # alpha = self.lasso.alpha_ * len(self.signal) * 2

    # Model to simulate from
    class SMCBridge(ssp.TemperingBridge):
        # mallocs for calculations
        abs_malloc = np.zeros((nboot, len(mu)), dtype=float)
        sum_malloc = np.zeros(nboot, dtype=float)
        exp_dist = stats.expon(scale=1/alpha)
        norm_dist = stats.multivariate_normal(mean=mu, cov=S)

        # model-specific implementation of log-density to sample from
        def logtarget(self, theta):
            np.abs(theta, out=self.abs_malloc)
            np.sum(self.abs_malloc, axis=1, out=self.sum_malloc)
            return 0.5*self.exp_dist.logpdf(self.sum_malloc) + self.norm_dist.logpdf(theta)
        
        # particles package doesn't normally get you to redefine this,
        # but here, the "prior" is the expensive calculation so we want to
        # avoid it; the default implementaion is loglik(theta) = logtarget(theta) - prior.logpdf(theta)
        def loglik(self, theta):
            np.abs(theta, out=self.abs_malloc)
            np.sum(self.abs_malloc, axis=1, out=self.sum_malloc)
            return 0.5*self.exp_dist.logpdf(self.sum_malloc)

    base_dist = dists.MvNormal(loc=mu, cov=S)
    smc_bridge = SMCBridge(base_dist=base_dist)
    print("Sampling...")
    # tempering_model = ssp.AdaptiveTempering(model=smc_bridge, len_chain=mcmc_len, wastefree=False)
    exponents = np.linspace(0, 1, num=n_smc, endpoint=True)
    tempering_model = ssp.Tempering(model=smc_bridge, len_chain=mcmc_len, wastefree=False, exponents=exponents)
    alg = particles.SMC(fk=tempering_model, N=nboot, verbose=True, store_history=True)
    alg.run()
    beta = alg.X.theta
    return beta.T, alg

import denoising

N = 128
t = np.linspace(0, 1, N)
y = np.sin(2*np.pi*t*8)
D = denoising.Dictionary(N, np.arange(1,N//5+1), np.ones(N//5))
X = D.dictionary
X.shape
lam = 1
n_smc=1000
betas, alg = simulate_posterior_fast(32, 8*1024, y, X, None, lam, 16, n_smc)
evidence = [sum(np.exp(alg.hist.wgts[i].lw)) for i in range(n_smc)]

import matplotlib.pyplot as plt

W = np.linalg.inv(np.dot(X.T, X) + (lam*np.eye(X.shape[1])))
mu = (W @ X.T) @ y

plt.figure()
plt.plot(np.log(evidence))
plt.show()

plt.figure()
plt.plot(X@betas, color="grey", alpha=0.01)
plt.plot(X @ mu)
plt.plot(y)
plt.show()
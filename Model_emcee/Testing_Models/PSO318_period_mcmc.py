import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from astropy.table import Table
import emcee
import corner
import scipy.optimize as op
import IPython

#import math 
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches 
from astropy.io import fits
#%matplotlib inline

# read in data
#data = Table.read('extraction_7pix/spitzer_lightcurve_JD.fits')
#data = Table.read('PSO-318results_newJD.fits')
data = Table.read('PSO318_Spitzer_unbinned.fits')

JDinit =  5.763948648921E+04
#t = (data['TIME'][0] - JDinit) * 24.
t = data['TIME'][0]
flux = data['FLUX'][0]

#t = t[:,0]
flux = flux[:,0]
fluxerr = np.zeros_like(flux) + np.std(flux - np.roll(flux, 1))


# try least squares fit first
guess_mean = np.mean(flux)
guess_std = 3*np.std(flux)/(2**0.5)
guess_phase = 0
guess_period = 8.0

# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(((t)/x[2])*2.*3.14159 + x[1]) + x[3] - flux
est_std, est_phase, est_period, est_mean = op.leastsq(optimize_func, [guess_std, guess_phase, guess_period, guess_mean])[0]



# set up MCMC

# parameters are: amplitude, period, phase, and offset

# define likelihood
def lnlike(theta, t, flux, fluxerr):
    amp, period, phase, offset = theta
    model = amp * np.sin((t/period)*2.*3.14159 + phase) + offset
    inv_sigma2 = 1.0/(fluxerr**2) # + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((flux-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# maximize likelihood to determine first guess
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [est_std, est_period, est_phase, est_mean], args=(t, flux, fluxerr))
amp_ml, period_ml, phase_ml, offset_ml = result["x"]

# define priors
def lnprior(theta):
    amp, period, phase, offset = theta
    if 0 < amp < 1.0 and 0.0 < period < 100.0 and -2.*3.14159 < phase < 2.*3.14159 and 0 < offset < 2.0:
        return 0.0
    return -np.inf

# define posterior
def lnprob(theta, t, flux, fluxerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, t, flux, fluxerr)

print "running MCMC"
#initialize MCMC
ndim, nwalkers = 4, 1000
pos = [ [est_std, est_period, est_phase, est_mean] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#IPython.embed()

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux, fluxerr))

# run the MCMC for 500 steps starting from the tiny ball defined above:
sampler.run_mcmc(pos, 4000)


samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
samples[:, 0] = samples[:, 0] * 100.

fig = corner.corner(samples, labels=["$amplitude$", "$period (hr)$", "$phase$", "$mean$"], quantiles = [0.16, 0.5, 0.84], show_titles=True, labels_args={"fontsize":40}, title_fmt=".3f")

#fig = corner.corner(samples, labels=["$amplitude$", "$period$", "$phase$", "$mean$"])
fig.savefig("PSO318_period_triangle_v9.png")


for amp, period, phase, offset in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(t, amp * np.sin((t/period)*2.*3.14159 + phase) + offset, color="k", alpha=0.1)
plt.plot(t, flux, color="r", lw=2, alpha=0.8)

fits.writeto('PSO318_sampler_fixed_newtime_unbinned.fits', sampler.chain)

fits.writeto('PSO318_flat_chain_fixed_newtime_unbinned.fits', samples)


amp_mcmc, period_mcmc, phase_mcmc, offset_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))


amp_mcmc95, period_mcmc95, phase_mcmc95, offset_mcmc95 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [2.5, 50, 97.5],
                                                axis=0)))



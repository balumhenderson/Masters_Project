#####__Trial for 1 spot and 2 bands__#####
import numpy as np
from astropy.table import Table
import SpotBandModSingle
import emcee
import scipy.optimize as op
import corner
import matplotlib.pyplot as pl

#Read in the observed data
obsDatName = "out.fits"
#obsDatName = input("What is the observed data file name?\n")
obsDat = Table.read(obsDatName, format = "fits")

#####__Values that will be used in the calculations__#####


#Define the number of variables, walkers and the number of points
ndim, nwalkers, points = 3, 400, 200


#Observed body variables
bodyDiam = 139822000.0
rotPer = 1.0
incl = 90.0

#System variables
timeStep = 1/48
numDays = 20
t = obsDat["Time"]
flux = obsDat["Flux"]
fluxerr = np.zeros_like(flux) + np.std(flux - np.roll(flux, 1))


#Band1 Variables
sb2 = "Band"
guess_sizeB1 = 0.1*bodyDiam
guess_latB1 = 0.0
guess_phaseB1 = 0.0
guess_relBrightB1 = 1.5
guess_relVelB1 = 1.0


#Define arrays that will be held in the table
sb = np.array([sb2])
size = np.array([guess_sizeB1])
lat = np.array([guess_latB1])
phase = np.array([guess_phaseB1])
relBright = np.array([guess_relBrightB1])
relVel = np.array([guess_relVelB1])

#Define the table used to store the data
guess_table = Table([sb, size, relBright, relVel, lat, phase], names = ("Spot/Band","Size", "RelBright", "RelVel", "Latitude", "Phase"))
modTable = guess_table




#####__Start of the Calculations__#####

#Find the output from the initial guesses
outputTable = SpotBandModSingle.spotband(guess_table, bodyDiam, rotPer, incl, timeStep, numDays)



#####__MCMC__#####

print(modTable, end="\r")

nll = lambda *args: SpotBandModSingle.lnlike(*args)
result = op.minimize(nll, [modTable["RelBright"], modTable["RelVel"], modTable["Phase"]], args = (flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays))

#print(result["x"])
#print(modTable)

pos = np.array([result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, SpotBandModSingle.lnprob, args = (flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays))
sampler.run_mcmc(pos, points)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

mcmc_table = modTable
samT = Table(samples)
#outFile = open("outTable.txt", "w")
#samT.write(outFile, format="ascii.fixed_width")

print("\n\nPlotting...\n")

fig = corner.corner(samples, labels=["$Band 1 Velocity$", "$Band 1 Brightness$", "$Band 1 Phase$"], quantiles = [0.16, 0.5, 0.84], show_titles=True, labels_args={"fontsize":40}, title_fmt=".3f")

fig.savefig("Singlecorner.png")

print("Mean acceptance fraction: {0:.3f}" .format(np.mean(sampler.acceptance_fraction)))

print("\nAll Done!")


#for i in range(2):
#	mcmc_table["RelBright"], mcmc_table["RelVel"], mcmc_table["Phase"] = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))











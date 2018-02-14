#####__Trial for 1 spot and 2 bands__#####
import numpy as np
from astropy.table import Table
import SpotBandMod
import emcee
import scipy.optimize as op
import corner
import matplotlib.pyplot as pl

#Read in the observed data
obsDatName = "out.fits"
#obsDatName = input("What is the observed data file name?\n")
obsDat = Table.read(obsDatName, format = "fits")

#####__Values that will be used in the calculations__#####

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

#Spot variables
sb1 = "Spot"
guess_sizeS = 0.02*bodyDiam
guess_minorDiam = (2/3)*guess_sizeS
guess_latS = 0.0
guess_phaseS = 0.0
guess_relBrightS = 1.5
guess_relVelS = 1.5

#Band1 Variables
sb2 = "Band"
guess_sizeB1 = 0.1*bodyDiam
guess_latB1 = 0.0
guess_phaseB1 = 0.0
guess_relBrightB1 = 1.5
guess_relVelB1 = 1.5

#Band2 Variables
sb3 = "Band"
guess_sizeB2 = 0.1*bodyDiam
guess_latB2 = 0.0
guess_phaseB2 = 0.0
guess_relBrightB2 = 1.5
guess_relVelB2 = 1.5

#Define arrays that will be held in the table
sb = np.array([sb1, sb2, sb3])
size = np.array([guess_sizeS, guess_sizeB1, guess_sizeB2])
lat = np.array([guess_latS, guess_latB1, guess_latB2])
phase = np.array([guess_phaseS, guess_phaseB1, guess_phaseB2])
relBright = np.array([guess_relBrightS, guess_relBrightB1, guess_relBrightB2])
relVel = np.array([guess_relVelS, guess_relVelB1, guess_relVelB2])

#Define the table used to store the data
guess_table = Table([sb, size, relBright, relVel, lat, phase], names = ("Spot/Band","Size", "RelBright", "RelVel", "Latitude", "Phase"))
modTable = guess_table
#modTable["Size"][:] = 0.0
#modTable["RelBright"][:] = 0.0
#modTable["RelVel"][:] = 0.0
#modTable["Latitude"][:] = 0.0
#modTable["Phase"][:] = 0.0

#####__Start of the Calculations__#####

#Find the output from the initial guesses
outputTable = SpotBandMod.spotband(guess_table, bodyDiam, rotPer, incl, timeStep, numDays)



#####__MCMC__#####

nll = lambda *args: SpotBandMod.lnlike(*args)
result = op.minimize(nll, [modTable["RelBright"], modTable["RelVel"], modTable["Phase"]], args = (flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays))

#print(result["x"])
#print(modTable)

ndim, nwalkers = 9, 400

pos = np.array([result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, SpotBandMod.lnprob, args = (flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays))
sampler.run_mcmc(pos, 2000)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))


mcmc_table = modTable
samT = Table(samples)
#outFile = open("outTable.txt", "w")
#samT.write(outFile, format="ascii.fixed_width")


fig = corner.corner(samples, labels=["$Spot Velocity$", "$Band 1 Velocity$", "$Band 2 Velocity$", "$Spot Brightness$", "$Band 1 Brightness$", "$Band 2 Brightness$", "$Spot Phase$", "$Band 1 Phase$", "$Band 2 Phase$"], quantiles = [0.16, 0.5, 0.84], show_titles=True, labels_args={"fontsize":40}, title_fmt=".3f")

fig.savefig("corner.png")



#for i in range(2):
#	mcmc_table["RelBright"], mcmc_table["RelVel"], mcmc_table["Phase"] = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))











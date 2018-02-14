
#Arguments as: .txt file name with no extention, body diameter in metres, body rotational period in days,
#inclination in degrees, timeStep in fractions of days and numDays as an integer.

import numpy as np


def spotband(t, bodyDiam, bodyRotPer, incl, timeStep, numDays):
	from astropy.table import Table
	import numpy as np


	totArea = 2*np.pi*bodyDiam
	incl = incl*np.pi/180

	t["Latitude"] = t["Latitude"]*np.pi/180
	t["Phase"] = t["Phase"]*np.pi/180
	rotVel = np.array(t["RelVel"]*(2*np.pi/(bodyRotPer)))
	time = timeStep*np.arange(int(numDays/timeStep))
	spotTime = time
	bandTime = time
	relBright = t["RelBright"]
	for i in range(len(t)):
		if (0.0 < t["RelBright"][i] < 1.0):
			relBright[i] = (-1.0)/t["RelBright"][i]
		else:
			relBright[i] = t["RelBright"][i]-1


	spotPos = np.ones(int(numDays/timeStep))
	spotLat = np.ones(int(numDays/timeStep))
	spotBright = np.ones(int(numDays/timeStep))
	spotArea = np.ones(int(numDays/timeStep))
	spotPhase = np.ones(int(numDays/timeStep))
	bandPos = np.ones(int(numDays/timeStep))
	bandSectPos = np.ones(int(numDays/timeStep))
	bandLat = np.ones(int(numDays/timeStep))
	bandPhase = np.ones(int(numDays/timeStep))
	bandArea = np.ones(int(numDays/timeStep))
	bandBright = np.ones(int(numDays/timeStep))
	spotNo = 0
	bandNo = 0


	for i in range(len(t)):

		#The calculations relating only to spots.
		if t["Spot/Band"][i] == "Spot":
			spotNo = ++1
			#Add a column to the spotTime array for each spot that is read in.
			spotTime = np.c_[spotTime, time] 
			#Add a column to the spotPos array with values of each spot's rotational velocity.
			spotPos = np.c_[spotPos, (rotVel[i]*np.ones(int(numDays/timeStep)))]
			#Add a column to the spotPhase array with values of each spot's initial phase difference.
			spotPhase = np.c_[spotPhase, t["Phase"][i]*np.ones(int(numDays/timeStep))]
			#Add a column to the spotLat array with values of each spot's latitude.
			spotLat = np.c_[spotLat, (t["Latitude"][i]*np.ones(int(numDays/timeStep)))]
			#Add a column to the spotBright array with values of each spot's brightness.
			spotBright = np.c_[spotBright, relBright[i]*np.ones(int(numDays/timeStep))]
			#Add a column to the spotArea array with values of each spot's area.
			spotArea = np.c_[spotArea, (np.pi*(2/3)*t["Size"][i]*t["Size"][i]/4)*np.ones(int(numDays/timeStep))]


		#The calculations relating only to bands.
		elif t["Spot/Band"][i] == "Band":
			bandNo = ++1
			#Add a column to the bandTime array for each band that is read in.
			bandTime = np.c_[bandTime, time]
			#Add a column to the bandPos array with values of each band's rotational velocity.
			bandPos = np.c_[bandPos, (rotVel[i]*np.ones(int(numDays/timeStep)))]
			#Add a column to the bandLat array with values of each band's latitude.
			bandLat = np.c_[bandLat, (t["Latitude"][i]*np.ones(int(numDays/timeStep)))]
			#Add a column to the bandPhase array with values of each band's initial phase difference.
			bandPhase = np.c_[bandPhase, t["Phase"][i]*np.ones(int(numDays/timeStep))]
			#Add a column to the bandArea array with values of each band's area.
			bandArea = np.c_[bandArea, (t["Size"][i]*np.pi*bodyDiam*np.cos(t["Latitude"][i]))*np.ones(int(numDays/timeStep))]
			#Add a column to the bandBright array with values of each band's brightness amplitude.
			bandBright = np.c_[bandBright, (relBright[i]*np.ones(int(numDays/timeStep)))]
	
		else:
			pass

	if (spotNo > 0):
		#Calculate the output for the spots.
		spotPos = (spotPos*spotTime)+spotPhase
		spotXProj = np.sin(incl)*np.cos(spotPos)*np.cos(spotLat)+np.cos(incl)*np.sin(spotPos)*np.sin(spotLat)
		#Set any negative x-projections to zero, so that the spot is not seen when on the far side.
		spotXProj = spotXProj.clip(min=0)
		#Delete the first entry of the necessary arrays which was just a placeholder.
		spotXProj = np.delete(spotXProj,0,1)
		spotBright = np.delete(spotBright,0,1)
		spotArea = np.delete(spotArea,0,1)
		#Calculate the brightness output from each spot, and then sum the outputs into one array.
		spotOutput = (spotXProj*spotBright*spotArea)/totArea
		spotOutputTot = spotOutput.sum(axis=1)
	else:
		spotOutputTot = np.zeros(int(numDays/timeStep))

	if (bandNo > 0):
		#Calculate the output for the bands.
		bandPos = (bandPos*bandTime) + bandPhase
		bandXProj = np.sin(incl)*np.cos(bandPos)*np.cos(bandLat)+np.cos(incl)*np.sin(bandPos)*np.sin(bandLat)
		#Delete the first entry of the necessary arrays which was just a placeholder.
		bandXProj = np.delete(bandXProj,0,1)
		bandBright = np.delete(bandBright,0,1)
		bandArea = np.delete(bandArea,0,1)
		bandLat = np.delete(bandLat,0,1)
		#Calculate the area of each band that is seen at any one time, setting negative values to zero.
		bandArea = bandArea*((0.5*np.sin(incl)*np.cos(bandLat)+np.cos(incl)*np.sin(bandLat)).clip(min=0))
		#Calculate the brightness output from each band, and then sum the outputs to one array.
		bandOutput = (bandXProj*bandBright*bandArea)/totArea
		bandOutputTot = bandOutput.sum(axis=1)
	else:
		bandOutputTot = np.zeros(int(numDays/timeStep))


	#Calculate the combined output for each simulated time point by summing the contents of the previous two.
	combOutputTot = (np.c_[spotOutputTot, bandOutputTot]).sum(axis=1)

	return Table([time, combOutputTot], names = ("Time", "Flux"))






def lnlike(theta, flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays):
	#print(modTable)
	#print(theta)
	modTable["RelBright"] = theta[:3]
	modTable["RelVel"] = theta[3:6]
	modTable["Phase"] = theta[6:]
	model = spotband(modTable, bodyDiam, rotPer, incl, timeStep, numDays)
	print(modTable, end="\r")
	inv_sigma2 = 1.0/(fluxerr**2)
	return -0.5*(np.sum((flux-model["Flux"])**2 * inv_sigma2 - np.log(inv_sigma2)))



def lnprior(theta):
    relBright = theta[:3]
    relVel = theta[3:6]
    phase = theta[6:]
    if 0.0 <= relBright.all() <= 2.5 and 0.1 <= relVel.all() <= 1.5 and -45.0 <= phase.all() <= 45.0:
        return 0.0
    return -np.inf


def lnprob(theta, flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, flux, fluxerr, modTable, bodyDiam, rotPer, incl, timeStep, numDays)














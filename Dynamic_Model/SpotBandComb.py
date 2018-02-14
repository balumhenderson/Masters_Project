import numpy as np
import matplotlib.pyplot as pl

from astropy.table import Table

import csv

#Define the variables about the system
bodyDiam = 139822000.0
totArea = 2*np.pi*(bodyDiam/2)**2
bodyBright = 1.0
bodyRotPer = 13.2
incl = 90.0*np.pi/180

#Read in the parameters of the spots and bands from the file.
t = Table.read("SpotBand.txt", format = "ascii.fixed_width")

#Create arrays that will hold all of the time information.
timeStep = 2.5
numStep = 5000
time = timeStep*np.arange(numStep)
spotTime = time
bandTime = time


#Change the read-in variables to radians, radians per minute.
t["Latitude"] = t["Latitude"]*np.pi/180
t["Phase"] = t["Phase"]*np.pi/180
#t["RelBright"] = (t["RelBright"]*bodyBright)-(t["RelBright"]+bodyBright)
rotVel = np.array(t["RelVel"]*((2*np.pi/bodyRotPer)/60))


#Define arrays, that will eventually hold the information for each spot/band for all time.
spotPos = np.ones(numStep)
spotLat = np.ones(numStep)
spotBright = np.ones(numStep)
spotArea = np.ones(numStep)
spotPhase = np.ones(numStep)
bandPos = np.ones(numStep)
bandSectPos = np.ones(numStep)
bandLat = np.ones(numStep)
bandPhase = np.ones(numStep)
bandArea = np.ones(numStep)
bandBright = np.ones(numStep)


#Go through each row in the input table and append the spot and band arrays.
for i in range(len(t)):

	#The calculations relating only to spots.
	if t["Spot/Band"][i] == "Spot":
		
		#Add a column to the spotTime array for each spot that is read in.
		spotTime = np.c_[spotTime, time] 
		#Add a column to the spotPos array with values of each spot's rotational velocity.
		spotPos = np.c_[spotPos, (rotVel[i]*np.ones(numStep))]
		#Add a column to the spotPhase array with values of each spot's initial phase difference.
		spotPhase = np.c_[spotPhase, t["Phase"][i]*np.ones(numStep)]
		#Add a column to the spotLat array with values of each spot's latitude.
		spotLat = np.c_[spotLat, (bodyBright*t["Latitude"][i]*np.ones(numStep))]
		#Add a column to the spotBright array with values of each spot's brightness.
		if t["Permanent"][i] == "True":
			spotBright = np.c_[spotBright, t["RelBright"][i]*np.ones(numStep)]
		else:
			spotBrightDyn = t["RelBright"][i]*np.sin(np.pi*time/(t["SpotEnd"][i]-t["SpotBegin"][i])).clip(min=0)
			spotBrightDyn[0:t["SpotBegin"][i]] = 0
			spotBrightDyn[t["SpotEnd"][i]:] = 0
			spotBright = np.c_[spotBright, spotBrightDyn]
		#Add a column to the spotArea array with values of each spot's area.
		spotArea = np.c_[spotArea, (np.pi*t["MajorDiam"][i]*t["MinorDiam"][i]/4)*np.ones(numStep)]


	#The calculations relating only to bands.
	elif t["Spot/Band"][i] == "Band":
		#Add a column to the bandTime array for each band that is read in.
		bandTime = np.c_[bandTime, time]
		#Add a column to the bandPos array with values of each band's rotational velocity.
		bandPos = np.c_[bandPos, (rotVel[i]*np.ones(numStep))]
		#Add a column to the bandLat array with values of each band's latitude.
		bandLat = np.c_[bandLat, (t["Latitude"][i]*np.ones(numStep))]
		#Add a column to the bandPhase array with values of each band's initial phase difference.
		bandPhase = np.c_[bandPhase, t["Phase"][i]*np.ones(numStep)]
		#Add a column to the bandArea array with values of each band's area.
		bandArea = np.c_[bandArea, (t["BandHeight"][i]*np.pi*bodyDiam*np.cos(t["Latitude"][i]))*np.ones(numStep)]
		#Add a column to the bandBright array with values of each band's brightness amplitude.
		bandBright = np.c_[bandBright, (t["RelBright"][i]*np.ones(numStep))]
	
	else:
		pass


#Calculate the output for the spots.
spotPos = (spotPos*spotTime)+spotPhase
spotXProj = np.sin(incl)*np.cos(spotPos)*np.cos(spotLat)+np.cos(incl)*np.sin(spotPos)*np.sin(spotLat)
#Set any negative x-projections to zero, so that the spot is not seen when on the far side.
spotXProj = spotXProj.clip(min=0)
#Delete the first column of the necessary arrays which was just a placeholder.
spotXProj = np.delete(spotXProj,0,1)
spotBright = np.delete(spotBright,0,1)
spotArea = np.delete(spotArea,0,1)
#Calculate the brightness output from each spot, and then sum the outputs into one array.
spotOutput = (spotXProj*spotBright*spotArea)/totArea
spotOutputTot = bodyBright+spotOutput.sum(axis=1)


#Calculate the output for the bands.
bandPos = (bandPos*bandTime) + bandPhase
bandXProj = np.sin(incl)*np.cos(bandPos)*np.cos(bandLat)+np.cos(incl)*np.sin(bandPos)*np.sin(bandLat)
#Delete the first column of the necessary arrays which was just a placeholder.
bandXProj = np.delete(bandXProj,0,1)
bandBright = np.delete(bandBright,0,1)
bandArea = np.delete(bandArea,0,1)
bandLat = np.delete(bandLat,0,1)
#Calculate the area of each band that is seen at any one time, setting negative values to zero.
bandArea = bandArea*((0.5*np.sin(incl)*np.cos(bandLat)+np.cos(incl)*np.sin(bandLat)).clip(min=0))
#Calculate the brightness output from each band, and then sum the outputs to one array.
bandOutput = (bandXProj*bandBright*bandArea)/totArea
bandOutputTot = bodyBright+bandOutput.sum(axis=1)


#Calculate the combined output for each simulated time point by summing the contents of the previous two.
combOutputTot = (np.c_[spotOutputTot, bandOutputTot]).sum(axis=1)-bodyBright


#Write out the calculated data for the spots, the bands and both combined.
o = Table([time,spotOutputTot,bandOutputTot,combOutputTot], names=("Time","Spots","Bands","Combined"))
o.write("SpotBandCombOut.txt", format="ascii.fixed_width", overwrite=True)


# Plot and display the simulation data for the spots, the bands and the combined output.
i = Table.read("SpotBandCombOut.txt", format = "ascii.fixed_width")

fig = pl.figure()

axSpot = fig.add_subplot(311)
axSpot.set_title('Spot Output')
axSpot.set_ylabel('Visibility / units')
axSpot.plot(i["Time"],i["Spots"])
axSpot.set_xticks([])

axBand = fig.add_subplot(312)
axBand.set_title('Band Output')
axBand.set_ylabel('Visibility / units')
axBand.plot(i["Time"],i["Bands"])
axBand.set_xticks([])

axComb = fig.add_subplot(313)
axComb.set_title('Comb Output')
axComb.set_xlabel('Time / min')
axComb.set_ylabel('Visibility / units')
axComb.plot(i["Time"],i["Combined"])

pl.ticklabel_format(useOffset=False)
pl.show()






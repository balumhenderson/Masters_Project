import numpy as np
import matplotlib.pyplot as pl
import os.path
import datetime
from astropy.table import Table
import csv

#Define the variables about the system.
bodyDiam = 139822000.0
totArea = 2*np.pi*(bodyDiam/2)**2
#bodyBright = 1.0
bodyRotPer = 1.0	# Rotational period in units of days.
incl = 90.0*np.pi/180

#Define the output folder path that will be used to store all the data.
nowT = datetime.datetime.now()
now = nowT.strftime("%d-%m-%Y_%H:%M:%S")
#savePath = os.path.join("output/", now)
#if not os.path.exists(savePath):
#	os.makedirs(savePath)
txtName = ("in.txt")
fitsName = ("out.fits")
pngName = ("out.png")
asciiName = ("out.txt")

#Write the .fits directory to a list in a .txt file.
#saveFile = open("Fits_Files.txt", "a+")
#saveFile.write(fitsName +"\n")
#saveFile.close()

#Read in the parameters of the spots and bands from the file.
t = Table.read("SpotBand.txt", format = "ascii.fixed_width")
#Write the input table to the output folder for future reference.
saveFile = open(txtName, "w+")
t.write(saveFile, format = "ascii.fixed_width")
saveFile.close()



#Create arrays that will hold all of the time information.
timeStep = 1/48		# Value of increments between calculations, in units of days
numStep = 20		# Number of days to run the simulation for
time = timeStep*np.arange(int(numStep/timeStep))
spotTime = time
bandTime = time
spotNo = 0
bandNo = 0


#Change the read-in variables to radians, radians per day.
t["Latitude"] = t["Latitude"]*np.pi/180
t["Phase"] = t["Phase"]*np.pi/180
for i in range(len(t)):
	if (t["RelBright"][i] < 1.00):
		t["RelBright"][i] = (-1)/t["RelBright"][i]
	else:
		t["RelBright"][i] = t["RelBright"][i]-1
rotVel = np.array(t["RelVel"]*(2*np.pi/(bodyRotPer)))


#Define arrays, that will eventually hold the information for each spot/band for all time.
spotPos = np.ones(int(numStep/timeStep))
spotLat = np.ones(int(numStep/timeStep))
spotBright = np.ones(int(numStep/timeStep))
spotArea = np.ones(int(numStep/timeStep))
spotPhase = np.ones(int(numStep/timeStep))
bandPos = np.ones(int(numStep/timeStep))
bandSectPos = np.ones(int(numStep/timeStep))
bandLat = np.ones(int(numStep/timeStep))
bandPhase = np.ones(int(numStep/timeStep))
bandArea = np.ones(int(numStep/timeStep))
bandBright = np.ones(int(numStep/timeStep))

col2 = np.zeros(int(numStep/timeStep))
col3 = np.zeros(int(numStep/timeStep))
col4 = np.zeros(int(numStep/timeStep))
col5 = np.zeros(int(numStep/timeStep))
col7 = np.zeros(int(numStep/timeStep))
col8 = np.zeros(int(numStep/timeStep))
col9 = np.zeros(int(numStep/timeStep))
col10 = np.zeros(int(numStep/timeStep))



#Go through each row in the input table and append the spot and band arrays.
for i in range(len(t)):

	#The calculations relating only to spots.
	if t["Spot/Band"][i] == "Spot":
		spotNo = ++1
		#Add a column to the spotTime array for each spot that is read in.
		spotTime = np.c_[spotTime, time] 
		#Add a column to the spotPos array with values of each spot's rotational velocity.
		spotPos = np.c_[spotPos, (rotVel[i]*np.ones(int(numStep/timeStep)))]
		#Add a column to the spotPhase array with values of each spot's initial phase difference.
		spotPhase = np.c_[spotPhase, t["Phase"][i]*np.ones(int(numStep/timeStep))]
		#Add a column to the spotLat array with values of each spot's latitude.
		spotLat = np.c_[spotLat, (t["Latitude"][i]*np.ones(int(numStep/timeStep)))]
		#Add a column to the spotBright array with values of each spot's brightness.
		spotBright = np.c_[spotBright, t["RelBright"][i]*np.ones(int(numStep/timeStep))]


#		if t["Permanent"][i] == "True":
#			spotBright = np.c_[spotBright, t["RelBright"][i]*np.ones(int(numStep/timeStep))]
#		else:
#			spotBrightDyn = t["RelBright"][i]*np.sin(np.pi*time/(t["SpotEnd"][i]-t["SpotBegin"][i])).clip(min=0)
#			spotBrightDyn[0:t["SpotBegin"][i]] = 0
#			spotBrightDyn[t["SpotEnd"][i]:] = 0
#			spotBright = np.c_[spotBright, spotBrightDyn]
		#Add a column to the spotArea array with values of each spot's area.
		spotArea = np.c_[spotArea, (np.pi*(t["Size"][i]**2)*2/3)*np.ones(int(numStep/timeStep))]


	#The calculations relating only to bands.
	elif t["Spot/Band"][i] == "Band":
		bandNo = ++1
		#Add a column to the bandTime array for each band that is read in.
		bandTime = np.c_[bandTime, time]
		#Add a column to the bandPos array with values of each band's rotational velocity.
		bandPos = np.c_[bandPos, (rotVel[i]*np.ones(int(numStep/timeStep)))]
		#Add a column to the bandLat array with values of each band's latitude.
		bandLat = np.c_[bandLat, (t["Latitude"][i]*np.ones(int(numStep/timeStep)))]
		#Add a column to the bandPhase array with values of each band's initial phase difference.
		bandPhase = np.c_[bandPhase, t["Phase"][i]*np.ones(int(numStep/timeStep))]
		#Add a column to the bandArea array with values of each band's area.
		bandArea = np.c_[bandArea, (t["Size"][i]*np.pi*bodyDiam*np.cos(t["Latitude"][i]))*np.ones(int(numStep/timeStep))]
		#Add a column to the bandBright array with values of each band's brightness amplitude.
		bandBright = np.c_[bandBright, (t["RelBright"][i]*np.ones(int(numStep/timeStep)))]
	
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
    spotOutputTot = np.zeros(int(numStep/timeStep))

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
    bandOutputTot = np.zeros(int(numStep/timeStep))


#Calculate the combined output for each simulated time point by summing the contents of the previous two.
combOutputTot = (np.c_[spotOutputTot, bandOutputTot]).sum(axis=1)


#Write out the calculated data for the time, combined output and the other columns necessary for the wavelet code.
o = Table([time,combOutputTot], names=("Time","Flux"))
saveFile = open(fitsName, "w+")
o.write(saveFile, format="fits", overwrite=True)
saveFile.close()


checkOut = Table([time, spotOutputTot, bandOutputTot, combOutputTot], names = ("Time","Spots","Bands","Flux"))
saveFile = open(asciiName, "w+")
checkOut.write(saveFile, format="ascii.fixed_width")
saveFile.close()

# Plot and display the simulation data for the spots, the bands and the combined output.
i = Table.read(asciiName, format = "ascii.fixed_width")

fig = pl.figure()

axSpot = fig.add_subplot(311)
axSpot.set_title('Spot Output')
axSpot.set_ylabel('Visibility / units')
axSpot.plot(i["Time"],i["Spots"], linewidth=0.6)
axSpot.set_xticks([])

axBand = fig.add_subplot(312)
axBand.set_title('Band Output')
axBand.set_ylabel('Visibility / units')
axBand.plot(i["Time"],i["Bands"], linewidth=0.6)
axBand.set_xticks([])

axComb = fig.add_subplot(313)
axComb.set_title('Comb Output')
axComb.set_xlabel('Time / Days')
axComb.set_ylabel('Visibility / units')
axComb.plot(i["Time"],i["Flux"], linewidth=0.6)


pl.ticklabel_format(useOffset=False)
fig.savefig(pngName, dpi = 600)
pl.show()






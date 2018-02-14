import numpy as np
from waveletFuncs import *
import matplotlib.pylab as plt
import matplotlib.patches as pchs
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from astropy.table import Table
from astropy.io import fits

#--------------------------------------------------------------------------
__author__ = 'Evgeniya Predybaylo'
# Original code: See "http://paos.colorado.edu/research/wavelets/"
# The Matlab code written January 1998 by C. Torrence is modified to Python by 
# Evgeniya Predybaylo, December 2014
#--------------------------------------------------------------------------

#=========================================================
# READ THE KEPLER DATA
#=========================================================


timeunit = 'BJD - 2454833 (barycenter corrected JD)'
dataunit = 'e-/s (electrons/s)'

def nan_helper(y):
	return np.isnan(y), lambda z: z.nonzero()[0]

with open(sys.argv[1]) as f:			# opening .txt or .dat file to read list of .fits files
	files = f.readlines()
	
files = [x.strip() for x in files]

for name in files:
	print(name) 				#print out the filename we are currently processing
	
#	with open(name) as f:			#for .txt or .dat files
#		co = f.readlines()
#	co = [x.strip() for x in co]
#	co = np.array([float(x) for x in co])
#	time = np.arange(len(co))*1800/(3600*24)	#fake time series of correct interval, if needed

	hdulist = fits.open(name)		#open the .fits file

	data = hdulist[1].data
	time = data['time']			#set correct columns to time and flux:
	co = data['flux']			#SC: time, flux. EVEREST: TIME, FCOR. Raw: TIME, SAP_FLUX.

	#co = np.nan_to_num(co)
	dt   = time[1]-time[0]
	n = len(co) 				# the length of the array

		
#================================= Sigma clipping multiple times, normalising ================================
	
	for i in range(3):
		print(i+1)
		print(np.nanmean(co), np.sqrt(np.nanvar(co)))

		for ROW, VALUE in enumerate(co):				# sigma clipping below/above z*sigma
			if VALUE > (np.nanmean(co) + (3*np.sqrt(np.nanvar(co)))) or VALUE < (np.nanmean(co) - (3*np.sqrt(np.nanvar(co)))) or VALUE == math.nan:
				co[ROW] = math.nan

		if np.isnan(co).any() == True:					# interpolating across NaNs
			nans, nan_helper_output = nan_helper(co)
			co[nans]= np.interp(nan_helper_output(nans), nan_helper_output(~nans), co[~nans])

	#if min(co) <= 0:							# code does not like negative values
	#	for row, value in enumerate(co):
	#		co[row] = co[row] + 100

	avgval = np.mean(co)							# normalising to 1
	for row, value in enumerate(co):
		co[row] = co[row]/avgval




#================================================================================
# Wavelet 
#================================================================================

	variance = np.std(co, ddof=1) ** 2
	xlim = (time[0], time[-1])  # plotting range
	pad = 1  # pad the time series with zeroes (recommended)
	dj = 0.01  # this will do 4 sub-octaves per octave --- CAN VARY 
	s0 = 0.1 * dt  # this says start at a scale of 1/2*dt ---- CAN VARY
	j1 = 21 / dj  # this says do 7 powers-of-two with dj sub-octaves each ---- CAN VARY
	lag1 = 0.72  # lag-1 autocorrelation for red noise background
	mother = 'MORLET' 

# Wavelet transform: 
#
	wave, power, period, scale, coi = wavelet(co, dt, pad, dj, s0, j1, mother)
	coi = coi[:-1] # remove trailing zero


# Significance levels: (variance=1 for the normalized co)
	signif = wave_signif(([1.0]), dt=dt, sigtest=0, scale=scale, lag1=lag1, mother=mother)
	sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  
	sig95 = power / sig95  # where ratio > 1, power is significant
	sig95 = (sig95 - np.mean(sig95))/np.std(sig95, ddof = 1)**2
	sig95 = sig95*variance + np.mean(co)


# Global wavelet spectrum & significance levels:
	global_ws = variance * (np.sum(power, axis=1) / n)  # time-average over all times
	dof = n - scale  # the -scale corrects for padding at edges
	global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, lag1=lag1, dof=dof, mother=mother)


# Reconstruction: takes the output from the  wavelet analysis from above, and reconstructs the time series.
	reconstruction, discrepancy = reconstruct(co, wave, period, scale, dt, dj)


#===================== writing period and power to a .fits table =================================

#	col1 = fits.Column(name = 'period', format = 'D', array = period)
#	col2 = fits.Column(name = 'global_ws', format = 'D', array = global_ws)
#
#	cols = fits.ColDefs([col1, col2])
#	tbhdu = fits.BinTableHDU.from_columns(cols)
#	hdu = fits.PrimaryHDU()
#	thdulist = fits.HDUList([hdu, tbhdu])
#	tbhdu.writeto(name.replace('k2sc.fits', 'output.fits'))


#====================================================================================================



#------------------------------------------------------ Plotting------------------------------------------------------#
 
#FIGURE1
#
#a) the original time series, with the reconstructed time series plotted over the top. 
#Ideally, you shouldn't be able to distinguish the lines
#
#b) Comparison between the original and the de- and re-constructed series. 
#Left hand yaxis shows this as a percentage difference, the right as the absolute difference in ppb.
#
#c) wavelet power spectrum 
#
#d) global wavelet spectrum


#--- Plot time series
	fig1 = plt.figure(figsize=(18, 9))
	ax1 = fig1.add_subplot(221)
	ax2 = fig1.add_subplot(222)
	ax3 = fig1.add_subplot(223)
	ax4 = fig1.add_subplot(224)

# plot original data and reconstruction
	ax1.plot(time, co, time, reconstruction, linewidth=0.6)
	ax1.set_xlim(xlim[:])
	ax1.set_xlabel('Time '+timeunit)
	ax1.set_ylabel('PDCSAP_FLUX '+dataunit) 
	ax1.set_title('a) Aperture photometry flux ') #+ infile

# --- Plot percentage difference between the orignal and reconstructed series (left axis) and value of difference in ppb (right axis)
	ax2.plot(time, discrepancy, 'b', linewidth=0.6)
	ax2.set_xlim(xlim[:])
	ax2.set_xlabel('Time '+timeunit)
	ax2.set_ylabel('percentage difference (%)')
	ax2.set_title('d) Difference between original and reconstructed time series')
	
#--- Contour plot wavelet power spectrum
	minlvl = 0
	maxlvl = np.max(power)
	lvls = np.arange(minlvl, maxlvl, (maxlvl - minlvl)/50)
	ax3.contourf(time, period, power, levels = lvls ) 
	ax3.set_xlabel('Time '+timeunit)
	ax3.set_ylabel('Period (day)')
	ax3.set_title('b) PDCSAP_FLUX')
	ax3.set_xlim(xlim[:])
	ax3.set_ylim([0, 30])
	

# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
#	CS =ax3.contour(time, period, sig95, [-99, 1], colors='k', linewidth = 2)
	im = ax3.contourf(time, period, power, levels = lvls )	
#im = ax3.contourf(CS, levels=lvls)
# cone-of-influence, anything "below" is dubious
#	ax3.plot(time, coi, 'k')
	ax3.invert_yaxis()
# set up the size and location of the colorbar
	divider = make_axes_locatable(ax3)
	cax = divider.append_axes("bottom", size="5%", pad=0.5)
	cb = plt.colorbar(im, cax=cax, orientation='horizontal')
#	cb.set_clim(minlvl, maxlvl)
#	cb.set_label('power')
	
	
#--- Plot global wavelet spectrum
	ax4.plot(global_ws, period)
	ax4.plot(global_signif, period, '--')
	ax4.set_xlabel('Power ((e-/s)$^2$)')
	ax4.set_ylabel('Period (day)')
	ax4.set_title('c) Global Wavelet Spectrum')
	ax4.set_xlim([0, 1.25 * np.max(global_ws)])
# format y-scale
	ax4.set_ylim([0, 30])
	ax4.invert_yaxis()
	

#=============================== finding local maximum ==============================
	lwrlim = 0	
	uprlim = 1000	
	for row, value in enumerate (global_ws):		
		if value == np.max(global_ws[lwrlim:uprlim]):
			print(np.max(global_ws[lwrlim:uprlim]))
			max_period = '%.4f' % period[row]
			print(max_period)

# overplot a transparent box showing the range we looked at to find the maximum
	patchwidth = period[uprlim] - period[lwrlim]
	for p in [pchs.Rectangle( (-1, period[lwrlim]), 2, patchwidth, alpha=0.3, facecolor="#00ffff")]:
		ax4.add_patch(p)
# adding some text
		text(0.85, 0.9,'max: ' + str(max_period) + ' days', ha='center', va='center', size=16, transform=ax4.transAxes)

#=======================================================================================

	plt.tight_layout()
	fig1.savefig(name.replace('.fits', '_fig1.png'), bbox_inches='tight')
	plt.close()



#=========================================================
# Dominant variations
#=========================================================

#FIGURE2

#a) periods within the specified range (in this case 0.25-0.5
#days). Note that this takes the closest period available. The exact
#bounds are printed to the command line. Also given is the position in
#the array 'period' which contains this period. This is to make it easy
#to check how many periods were included in this range.
#
#eg.  >> lower is position: 14 at:  0.486981445723 and higher is
#position: 21 at: 1.63800380802 this means that the lower bound is
#slightly less than 0.5 years, and the upper slightly above 1.5
#years. This contains 21 - 14 = 7 discreet periods. The number of
#periods included is determined by the variables s0, dj, dt etc. at the
#top of the script
#
#Amplitude is plotted on the right have axis.
#
#b)low frequency oscillation: this is constructed from all the periods
#above the selected range 
# 
#c) high frequency oscillation: this is constructed from all the
#periods below the selected range



#define dominant period 
	lower = 0.2
	higher = 0.6

	seasonal, lowFreq, highFreq, specWave, specPeriod = pickFreq(co, wave, period, scale, dt, dj, lower, higher)

	fig2 = plt.figure(figsize =(18,9))

	ax5 = fig2.add_subplot(311)
	ax5a = ax5.twinx()
	ax6 = fig2.add_subplot(312)
	ax7 = fig2.add_subplot(313)


	ax5.plot(time, seasonal, label = 'Cycle', linewidth=0.6)
#ax5a.plot(time[locs], amps,'x-' , color = 'r', label= 'Amplitude')
#ax5.set_xlim(xlim[:])
#ax5.set_title('seasonal cycle (0.25-0.5 days)')
#ax5.set_ylabel('seasonal cycle (ppb)')
#ax5a.set_ylabel('Amplitude')
##ax5.set_ylim([0, 300])
#ax5.legend(loc = 2)
#ax5a.legend(loc = 1)

	ax6.plot(time, lowFreq, linewidth=0.6)
	ax6.set_xlim(xlim[:])
	ax6.set_ylabel('low frequency signal')
	ax6.set_title('low frequency (higher than 0.6 days))')
	
	ax7.plot(time, highFreq, linewidth=0.6)
	ax7.set_xlim(xlim[:])
	ax7.set_ylabel('high frequency signal')
	ax7.set_title('high frequency (less than 0.2 days))')
	ax7.set_xlabel('time (year)')

	fig2.savefig(name.replace('.fits', '_fig2.png'), bbox_inches='tight')
	plt.close()
	
sys.exit()

#print(np.shape(time),np.shape(seasonal))



# http://iopscience.iop.org/article/10.1088/0004-637X/779/2/172/meta
# photometric period of 8.9 hours (0.3702 days)

#window = 19

#tmp    = int(np.float(len(time))/window)
#lenind = window*tmp

#seasonal = seasonal[0:lenind]
#p = np.split(seasonal,window)
#print(np.shape(p))



#p = np.transpose(p)

#plt.matshow(p, aspect='auto')
#plt.ylabel('n*dt = 1 period')
#plt.xlabel('Successive periods')
#plt.colorbar()

#plt.show()
#sys.exit()




 

# Create another fig in which we cut the data in timesteps of the period (0.37015 days), (added on 17/3/17)
#splitting the array after every 19 cells, then subtracting by the mean of that range to normalise time (as it should go from 0.0 to 0.37015)

# yields time_split, list splits into sublists every 19 values
#h=0
#j=0
#newlist=[]
#time_split=[]
#for i in range(len(time)):
#    newlist.append(time[i])
#    j +=1
#    if j>18:
#        time_split.append(newlist)
#        j=0
#        newlist=[]

# convert time list to a numpy array with subarrays
#time_split = np.array(time_split)

# normalise time list, and shift it so each subarray begins at ~0
#for x in range(0, len(time_split)):
#                   time_split[x] = time_split[x]-np.mean(time_split[x]) + 0.183917534

#print(time_split[1])
# yields co_split, list splits into sublists every 19 values
#h=0
#j=0
#newlist=[]
#co_split=[]
#for i in range(len(co)):
#    newlist.append(co[i])
#    j +=1
#    if j>18:
#        co_split.append(newlist)
#        j=0
#        newlist=[]

# convert co split list to a numpy array with subarrays
#co_split = np.array(co_split)


# make flux values increase monotonically between each subarray by a constant 

#for x in range(0, len(co_split)):
#                   co_split[x] = co_split[x]+300*x
#'''
# and then plot the data stacked as time increases (added on 16/3/17)

#fig3 = plt.figure(figsize=(18, 9))

#for x in range (0, len(time_split)):
#    ax8 = fig3.add_subplot(111)

#    ax8.plot(time_split[x], co_split[x], linewidth=0.6)
#    ax8.set_xlim([0,100])
#    ax8.set_xlabel('8.9 hr Period')
#    ax8.set_ylabel('Flux, increasing between each curve by a constant of 300')
#    ax8.set_title('Stack')

#plt.show()
#'''
# end of code


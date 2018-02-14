import sys
from scipy.special._ufuncs import gammainc, gamma
import numpy as np
from pylab import *
from scipy.optimize import fminbound
__author__ = 'Evgeniya Predybaylo'
#modified and extended by Anna Mackie, also to include the getAmp function written by Eli Billauer, below

# Copyright (C) 1995-2004, Christopher Torrence and Gilbert P.Compo
# Python version of the code is written by Evgeniya Predybaylo in 2014
#
#   This software may be used, copied, or redistributed as long as it is not
#   sold and this copyright notice is reproduced on each copy made. This
#   routine is provided as is without any express or implied warranties
#   whatsoever.
#
# Notice: Please acknowledge the use of the above software in any publications:
#    ``Wavelet software was provided by C. Torrence and G. Compo,
#      and is available at URL: http://paos.colorado.edu/research/wavelets/''.
#
# Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
#            Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.
#
# Please send a copy of such publications to either C. Torrence or G. Compo:
#  Dr. Christopher Torrence               Dr. Gilbert P. Compo
#  Research Systems, Inc.                 Climate Diagnostics Center
#  4990 Pearl East Circle                 325 Broadway R/CDC1
#  Boulder, CO 80301, USA                 Boulder, CO 80305-3328, USA
#  E-mail: chris[AT]rsinc[DOT]com         E-mail: compo[AT]colorado[DOT]edu
#
#-------------------------------------------------------------------------------------------------------------------



#wavelet function, returns the wave matrix, the periods used, the scale and the COI - added 'power' and made it a returnable variable
def wavelet(Y, dt, pad=0, dj=-1, s0=-1, J1=-1, mother=-1, param=-1):

        n1 = len(Y)

        if s0 == -1: # these are default values - might want to change these later on
                s0 = 2 * dt
        if dj == -1:
                dj = 1. / 4.
        if J1 == -1:
                J1 = np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj)
        if mother == -1:
                mother = 'MORLET'

        #....construct time series to analyze, pad if necessary
        x = Y - np.mean(Y)
        if pad == 1:
                base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)  # power of 2 nearest to N
                x = np.concatenate((x, np.zeros(2 ** (int(base2) + 1) - n1)))

        n = len(x)

	#....construct wavenumber array used in transform [Eqn(5)]
        kplus = np.arange(1, np.fix(n / 2 + 1))
        kplus = (kplus * 2 * np.pi / (n * dt))
        kminus = (-(kplus[0:-1])[::-1])
        k = np.concatenate(([0.], kplus, kminus))

	#....compute FFT of the (padded) time series
        f = np.fft.fft(x)  # [Eqn(3)]

	#....construct SCALE array & empty PERIOD & WAVE arrays
        j = np.arange(0,J1+1)
        scale = s0 * 2. ** (j * dj)
        wave = np.zeros(shape=(int(J1) + 1, n), dtype=complex)  # define the wavelet array

	# loop through all scales and compute transform
        for a1 in range(0, int(J1+1)):
                daughter, fourier_factor, coi, dofmin = wave_bases(mother, k, scale[a1], param)
                wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform[Eqn(4)]
        period = fourier_factor * scale  #[Table(1)]
        coi = coi * dt * np.concatenate((np.insert(np.arange((n1 + 1) / 2 - 1), [0], [1E-5]),np.insert(np.flipud(np.arange(0, n1 / 2 - 1)), [-1], [1E-5])))  # COI [Sec.3g]
        wave = wave[:, :n1]  # get rid of padding before returning
        power = (np.abs(wave))**2 # wavelet power spec
        return wave, power, period, scale, coi

#wave_bases: unchanged

def wave_bases(mother, k, scale, param):
        n = len(k)
        kplus = np.array(k > 0., dtype=float)

        if mother == 'MORLET':  #-----------------------------------  Morlet

                if param == -1:
                        param = 6.

                k0 = np.copy(param)
                expnt = -(scale * k - k0) ** 2 / 2. * kplus
                norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)  # total energy=N   [Eqn(7)]
                daughter = norm * np.exp(expnt)
                daughter = daughter * kplus  # Heaviside step function
                fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))  # Scale-->Fourier [Sec.3h]
                coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
                dofmin = 2  # Degrees of freedom
        elif mother == 'PAUL':  #-------------------------------- Paul
                if param == -1:
                        param = 4.
                m = param
                expnt = -scale * k * kplus
                norm = np.sqrt(scale * k[1]) * (2 ** m / np.sqrt(m*np.prod(np.arange(1, (2 * m))))) * np.sqrt(n)
                daughter = norm * ((scale * k) ** m) * np.exp(expnt) * kplus
                fourier_factor = 4 * np.pi / (2 * m + 1)
                coi = fourier_factor * np.sqrt(2)
                dofmin = 2
        elif mother == 'DOG':  #--------------------------------  DOG
                if param == -1:
                        param = 2.
                m = param
                expnt = -(scale * k) ** 2 / 2.0
                norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
                daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
                fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
                coi = fourier_factor / np.sqrt(2)
                dofmin = 1
        else:
                print ('Mother must be one of MORLET, PAUL, DOG')

        return daughter, fourier_factor, coi, dofmin


def wave_signif(Y, dt, scale, sigtest=-1, lag1=-1, siglvl=-1, dof=-1, mother=-1, param=-1):
	n1 = len(np.atleast_1d(Y))
	J1 = len(scale) - 1
	s0 = np.min(scale)
	dj = np.log2(scale[1] / scale[0])

	if n1 == 1:
		variance = Y
	else:
		variance = np.std(Y) ** 2

	if sigtest == -1:
		sigtest = 0
	if lag1 == -1:
		lag1 = 0.0
	if siglvl == -1:
		siglvl = 0.95
	if mother == -1:
		mother = 'MORLET'

	# get the appropriate parameters [see Table(2)]
	if mother == 'MORLET':  #----------------------------------  Morlet
		empir = ([2., -1, -1, -1])
		if param == -1:
			param = 6.
			empir[1:] = ([0.776, 2.32, 0.60])
		k0 = param
		fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))  # Scale-->Fourier [Sec.3h]
	elif mother == 'PAUL':
		empir = ([2, -1, -1, -1])
		if param == -1:
			param = 4
			empir[1:] = ([1.132, 1.17, 1.5])
		m = param
		fourier_factor = (4 * np.pi) / (2 * m + 1)
	elif mother == 'DOG':  #-------------------------------------Paul
		empir = ([1., -1, -1, -1])
		if param == -1:
			param = 2.
			empir[1:] = ([3.541, 1.43, 1.4])
		elif param == 6:  #--------------------------------------DOG
			empir[1:] = ([1.966, 1.37, 0.97])
		m = param
		fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
	else:
		print ('Mother must be one of MORLET, PAUL, DOG')

	period = scale * fourier_factor
	dofmin = empir[0]  # Degrees of freedom with no smoothing
	Cdelta = empir[1]  # reconstruction factor
	gamma_fac = empir[2]  # time-decorrelation factor
	dj0 = empir[3]  # scale-decorrelation factor

	freq = dt / period  # normalized frequency
	fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)  # [Eqn(16)]
	fft_theor = variance * fft_theor  # include time-series variance
	signif = fft_theor
	if len(np.atleast_1d(dof)) == 1:
		if dof == -1:
			dof = dofmin

	if sigtest == 0:  # no smoothing, DOF=dofmin [Sec.4]
		dof = dofmin
		chisquare = chisquare_inv(siglvl, dof) / dof
		signif = fft_theor * chisquare  # [Eqn(18)]
	elif sigtest == 1:  # time-averaged significance
		if len(np.atleast_1d(dof)) == 1:
			dof = np.zeros(J1) + dof
		dof[dof < 1] = 1
		dof = dofmin * np.sqrt(1 + (dof * dt / gamma_fac / scale) ** 2)  # [Eqn(23)]
		dof[dof < dofmin] = dofmin   # minimum DOF is dofmin
		for a1 in range(0, J1 + 1):
			chisquare = chisquare_inv(siglvl, dof[a1]) / dof[a1]
			signif[a1] = fft_theor[a1] * chisquare
		print (chisquare)
	elif sigtest == 2:  # time-averaged significance
		if len(dof) != 2:
			print ('ERROR: DOF must be set to [S1,S2], the range of scale-averages')
		if Cdelta == -1:
			print ('ERROR: Cdelta & dj0 not defined for ' + mother + ' with param = ' + str(param))

		s1 = dof[0]
		s2 = dof[1]
		avg =  np.logical_and(scale >= 2, scale < 8)# scales between S1 & S2
		navg = np.sum(np.array(np.logical_and(scale >= 2, scale < 8), dtype=int))
		if navg == 0:
			print ('ERROR: No valid scales between ' + str(s1) + ' and ' + str(s2))
		Savg = 1. / np.sum(1. / scale[avg])  # [Eqn(25)]
		Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)  # power-of-two midpoint
		dof = (dofmin * navg * Savg / Smid) * np.sqrt(1 + (navg * dj / dj0) ** 2)  # [Eqn(28)]
		fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])  # [Eqn(27)]
		chisquare = chisquare_inv(siglvl, dof) / dof
		signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare  # [Eqn(26)]
	else:
		print ('ERROR: sigtest must be either 0, 1, or 2')

	return signif

#-------------------------------------------------------------------------------------------------------------------
# CHISQUARE_INV  Inverse of chi-square cumulative distribution function (cdf).
#
#   X = chisquare_inv(P,V) returns the inverse of chi-square cdf with V
#   degrees of freedom at fraction P.
#   This means that P*100 percent of the distribution lies between 0 and X.
#
#   To check, the answer should satisfy:   P==gammainc(X/2,V/2)

# Uses FMIN and CHISQUARE_SOLVE


def chisquare_inv(P, V):

	if (1 - P) < 1E-4:
		print ('P must be < 0.9999')

	if P == 0.95 and V == 2:  # this is a no-brainer
		X = 5.9915
		return X

	MINN = 0.01  # hopefully this is small enough
	MAXX = 1  # actually starts at 10 (see while loop below)
	X = 1
	TOLERANCE = 1E-4  # this should be accurate enough

	while (X + TOLERANCE) >= MAXX:  # should only need to loop thru once
		MAXX = MAXX * 10.
	# this calculates value for X, NORMALIZED by V
		X = fminbound(chisquare_solve, MINN, MAXX, args=(P,V), xtol=TOLERANCE )
		MINN = MAXX

	X = X * V  # put back in the goofy V factor

	return X  # end of code

#-------------------------------------------------------------------------------------------------------------------
# CHISQUARE_SOLVE  Internal function used by CHISQUARE_INV
	#
	#   PDIFF=chisquare_solve(XGUESS,P,V)  Given XGUESS, a percentile P,
	#   and degrees-of-freedom V, return the difference between
	#   calculated percentile and P.

	# Uses GAMMAINC
	#
	# Written January 1998 by C. Torrence

	# extra factor of V is necessary because X is Normalized


def chisquare_solve(XGUESS,P,V):

	PGUESS = gammainc(V/2, V*XGUESS/2)  # incomplete Gamma function

	PDIFF = np.abs(PGUESS - P)            # error in calculated P

	TOL = 1E-4
	if PGUESS >= 1-TOL:  # if P is very close to 1 (i.e. a bad guess)
		PDIFF = XGUESS   # then just assign some big number like XGUESS

	return PDIFF

#---------------- new functions added by Anna -------------- #

def reconstruct(ser,wave, period, scale, dt, dj):
        """
        Inputs:
        ser --- length n --- original series
        wave --- array size n x m, output from 'wavelet' --- complex array, has dimensions of the number of datapoints (n) and number of elements in periods (m)
        period --- length m, output from 'wavelet' --- array holding the periods of the scaled wavelets which get fitted to the series 
        scale --- length m, output from 'wavelet' --- array of scales applied to the wavelets (see T&C section 3f)
        dt --- number, user defined --- time step of the data
        dj --- number, user defined --- factor for scale averaging (see T&C section 3f)
        
        Outputs:
        reconst --- reconstructed series after wavelet analysis
        discrep --- percentage difference from original time series
        
        
        Function uses the output of the function 'wavelet' and reconstructs the time series so that variables can be adjusted to reduce differences from the original time series.
        
        
        """
        Cdelta = 0.7785 # constant, see T&C
        psi0 = np.pi**-0.25 # constant, see T&C
        n = len(wave[0,:])# number of data points
        m = len(period) # number of periods
        reconst = [0]*n 
        
        for i in range(0, n): # loop over data points
                for j in range(0, m): # loop over periods
                        reconst[i] = reconst[i] + (((dt**0.5)*dj)/(Cdelta*psi0))* wave[j,i].real/(scale[j]**0.5) # equation 13 from T&C
        # in order to compare to the original series, need to normalise reconst, and scale up to match the original series
        variance_recon = np.std(reconst, ddof = 1)**2
        reconst = (reconst - np.mean(reconst))/np.sqrt(variance_recon) # normalise reconst
        variance_orig = np.std(ser, ddof = 1)**2 # variance of original series
        reconst = reconst*np.sqrt(variance_orig) + np.mean(ser) # scale reconst up to original series
        discrep = ((reconst - ser)/ser)*100 # calculate percentage difference between the two series                                                                          
        

        return reconst, discrep

 
def pickFreq(ser, wave, period, scale, dt, dj, lower, higher):
        """
        Inputs:
        ser --- length n --- original series
        wave --- array size n x m, output from 'wavelet' --- complex array, has dimensions of the number of datapoints (n) and number of elements in periods (m)
        period --- length m, output from 'wavelet' --- array holding the periods of the scaled wavelets which get fitted to the series 
        scale --- length m, output from 'wavelet' --- array of scales applied to the wavelets (see T&C section 3f)
        dt --- number, user defined --- time step of the data
        dj --- number, user defined --- factor for scale averaging (see T&C section 3f)
        lower --- number, user defined --- lower bound of period for extraction
        higher --- number, user defined --- higher bound of period for extraction


        Outputs:
        pickedFreq --- series containing periods specified
        lowFreq --- series containing all periods longer (all freqs lower) than that specified
        highFreq --- series containing all periods shorter (all freqs higher) than that specified 
        specWave --- the elements of 'wave' which fall within the specified periods
        specPeriod --- the elements of 'period' which fall within the specfied periods
        
        
        Function uses the output of the function 'wavelet'. Effectively a band-pass filter: selects the specified periods (those between 'higher' and 'lower'), and produces a time series of just these frequencies. Also outputs the relevant parts of the 'wave' array and lists the periods which have been selected.

        NB: finds the closest period available to that specified, which may be longer or shorter than that specified.
        
        
        """
        l = np.argmin(np.abs(period - lower),axis = 0)
        h = np.argmin(np.abs(period - higher), axis = 0) +1
        print ('lower is position:', l, 'at: ', period[l], 'and higher is position:', h, 'at:', period[h])
        Cdelta = 0.778 # constant, see T&C
        psi0 = np.pi**-0.25 # constant, see T&C
        n = len(wave[0,:])
        m = len(period) 
        highFreq = [0]*n # make empty arrays
        pickedFreq = [0]*n 
        lowFreq =  [0]*n
        totFreq = [0]*n
        ### the specifed periods ###
        specWave = wave[l:h, :]
        specPeriod = period[l:h]
        for i in range(0, n): # loop over datapoints
                for j in range(l, h) : # loop over periods
                        pickedFreq[i] = pickedFreq[i] + (((dt**0.5)*dj)/(Cdelta*psi0))* wave[j,i].real/(scale[j]**0.5) # eq. 13 from T&C

        ### low frequencies/long periods ###
        for i in range(0, n):
                for j in range(h, m) : 
                        lowFreq[i] = lowFreq[i] + (((dt**0.5)*dj)/(Cdelta*psi0))* wave[j,i].real/(scale[j]**0.5)                        


        ### high frequencies/ short periods ###
        for i in range(0, n): 
                for j in range(0, l) : 
                        highFreq[i] = highFreq[i] + (((dt**0.5)*dj)/(Cdelta*psi0))* wave[j,i].real/(scale[j]**0.5)
        
        # in order to compare to the original series, need to normalise pickedFreq, lowFreq and highFreq and scale up to match the original series
        variance_orig = np.std(ser, ddof = 1)**2 # variance of original series

        variance_PF = np.std(pickedFreq, ddof = 1)**2 # variance of the pickedFreq series
        pickedFreq = (pickedFreq - np.mean(pickedFreq))/np.sqrt(variance_PF) 

        variance_LF = np.std(lowFreq, ddof = 1)**2
        lowFreq = (lowFreq - np.mean(lowFreq))/np.sqrt(variance_LF) # variance of the lowFreq series

        variance_HF = np.std(highFreq, ddof = 1)**2
        highFreq = (highFreq - np.mean(highFreq))/np.sqrt(variance_HF) # variance of the highFreq series
        


        # scale up to match the original series
        pickedFreq = pickedFreq*np.sqrt(variance_orig) + np.mean(ser) 
        lowFreq = lowFreq*np.sqrt(variance_orig) + np.mean(ser) 
        highFreq = highFreq*np.sqrt(variance_orig) + np.mean(ser) 


        return  pickedFreq, lowFreq, highFreq, specWave, specPeriod


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def getAmp(ser, delta):
        """
        Inputs:
        ser --- series want to extract amplitude from
        delta --- minimum difference from points either side of the maximum/minimum
        
        Outputs:
        amp --- amplitudes
        loc --- position of amplitudes
        
        
        Function finds the maxima/minima of the series, and gets the amplitude (difference of a min to the preceding max). Position is defined as half way between the max/min.

        NB: uses the function 'peakdet'

        NB2: this will only work with a reasonably smooth series (eg a seasonal cycle), there is no mechanism for discarding 'double' peaks
        """

        maxtab, mintab = peakdet(ser, delta)
        pk = maxtab[:,1]
        xpk = maxtab[:,0]
        n = len(pk)
        tr = mintab[:,1]
        xtr = mintab[:,0]
        m = len(tr)
        if xpk[0] == 0:
                pk = pk[1:]
                xpk =xpk[1:]
        if xtr[0] ==0:
                tr = tr[1:]
                xtr = xtr[1:]
        n = len(pk)
        m = len(tr)
        # find out the length of the number of max/min pairs 
        if n > m:
                nb = m
        elif m > n:
                nb = n
        elif n == m:
                nb = n

        amp = [0]*nb # empty arrays
        loc = [0]*nb
        if xpk[0] < xtr[0]: # if the first peak is a maximum
                for i in range(0,nb):
                        amp[i] = pk[i] - tr[i] # amplitude is difference
                        loc[i] = xpk[i] + (xtr[i] - xpk[i])/2 # location is half way between the max and min

        elif xpk[0] > xtr[0]: # if the first peak is a minimum
                tr = tr[1:] # discard first minimum
                xtr = xtr[1:]
                for i in range(0,len(xtr)):
                        amp[i] = pk[i] - tr[i]
                        loc[i] = xpk[i] + (xtr[i] - xpk[i])/2
        return amp, loc 


def xwt(X,Y, dt, pad, dj, s0, j1, mother):
        """
        Inputs:
        X --- first time series
        Y --- second time series
        dt --- time step
        dj --- factor for scale averaging (see T&C section 3f)
        s0 --- minimum value of scale (see T&C)
        j1 --- factor involved in determining scale (see T&C)
        mother --- type of wavelet used (usually Morlet)        

        Outputs:
        phase --- n x m array, where n is the number of time steps, and m is the number of periods fitted. Gives relative phase between the two series, at every point in time, at every period. 
        """
        if len(X) != len(Y):
                print ('series different lengths')
                
        waveX, powerX, periodX, scaleX, coiX = wavelet(X,dt, pad, dj, s0, j1, mother) #first series
        waveY, powerY, periodY, scaleY, coiY = wavelet(Y,dt, pad, dj, s0, j1, mother) # second
        n = len(waveX)
       

       #cross wavelet
        Wxy = waveX*np.conj(waveY)
       
       #phase
        phase = np.arctan2(np.imag(Wxy),np.real(Wxy))
       


        return phase

def specXwt(waveX, waveY):
        """
        as above, but use when you've already produced the wave arrays to avoid doing the wavelet analysis twice
        """
        Wxy = waveX*np.conj(waveY)
        phase = np.arctan2(np.imag(Wxy),np.real(Wxy))
        return phase

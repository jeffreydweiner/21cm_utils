import numpy as np
import pyfits as pyfits
#import ephem 
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import mlab
import time
import quaternion as quat
#import healpy as hp
#import numexpr
import scipy.interpolate as sp
#from simtools import *


# Define a local second in Julian time - useful for duration calculation
second_jul = 0.1 / 8640

# Define pi - useful for evaluation expressions
pi = np.pi
rtd=180/np.pi
dtr=np.pi/180
lowfreq=100
hifreq=200
stepfreq=.01
ri_start=150
ri_stop=160
fixed_az=60. #degrees
min_el=-60. #degrees
max_el=60.  #degrees
datalen=np.int(max_el-min_el) #just use integral number of degrees, 1 per step, for now

speclen=np.int((hifreq-lowfreq)/stepfreq) #generates the number of points used
eorsig=np.zeros(speclen) #generates array with 
eorfreq=np.arange(lowfreq,hifreq,stepfreq)

for f in eorfreq:
    if f<=ri_start:
        eorsig[f==eorfreq]=.15
    if f>ri_start:
        if f< ri_stop:
            eorsig[f==eorfreq]=.15-.15*(f-ri_start)/(ri_stop-ri_start)

components=loadtxt('components.dat')
#components is 5 values for each of 11 frequencies: freq, w1,w2,w3,w4
#here we make the interpolation functions to be called by freq later do the
#interpolation in loglog space- need to call it with log(freq) as argument
comp1function=sp.UnivariateSpline(log(components[:,0]),log(components[:,1]),s=0.0001)
comp2function=sp.UnivariateSpline(log(components[:,0]),(components[:,2]),s=.00001)
comp3function=sp.UnivariateSpline(log(components[:,0]),(components[:,3]),s=.00001,k=2)
comp4function=sp.UnivariateSpline(log(components[:,0]),(components[:,4]),s=.00001)


maps408=loadtxt('component_maps_408locked.dat')

manymaps32=[]

for i,freq in enumerate(eorfreq):
    manymaps32.append(makemap(freq,nside=32)+eorsig[i])


for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/4.0)

pixaz=zeros(datalen,dtype=np.integer)
for i,el in enumerate(arange(min_el,max_el)):
    pixaz[i]=hp.ang2pix(32,np.pi/2. - el*dtr,fixed_az*dtr) 
    
spec=zeros([speclen,datalen])
for j in range(speclen):
    for i,pix in enumerate(pixaz):
        spec[j,i]=manymaps32[j][pix]

 
#take the first difference along each of the returned spectra
dspec=copy(spec)
dspec[1:,:]=np.diff(spec,axis=0)
dspec[0,:]=dspec[1,:]
#fit to a power law (linear fit after the log)
dpowlawmodel=np.zeros([speclen,datalen])
for i in range(datalen):
    (ar,br)=polyfit(eorfreq,log(0.-dspec[:,i]),1)
    dpowlawmodel[:,i]=0.-e**polyval([ar,br],eorfreq)
    
powlawmodel=np.zeros([speclen,datalen])
for i in range(datalen):
    (ar,br,cr)=polyfit(eorfreq,log(spec[:,i]),2)
    powlawmodel[:,i]=e**polyval([ar,br,cr],eorfreq)
    
#noisy version of differnce data

#now lets make a white noise realization for each measurement. I've used 10kHz bins, if we stack 10000 seconds
#of measurements and assume 100% duty cycle, we  then get to assume 100 S integration time. Assuming Tsys is dominated by the 
#sky signal we expect RMS per amplitude spectrum bin to be Tsky/sqrt(1e4*10000)= Tsky/10000 nice round numbers.

specnoise=np.zeros([speclen,datalen])
for j in range(speclen):
    for i in range(datalen):
        specnoise[j,i]=spec[j,i]*randn()/10000.


dspecnoisy=copy(spec+specnoise)
dspecnoisy[1:,:]=np.diff(spec+specnoise,axis=0)
dspecnoisy[0,:]=dspecnoisy[1,:]
#fit to a power law (linear fit after the log)
dpowlawmodelnoisy=np.zeros([speclen,datalen])
for i in range(datalen):
    (ar,br)=polyfit(eorfreq,log(0.-dspecnoisy[:,i]),1)
    dpowlawmodelnoisy[:,i]=0.-e**polyval([ar,br],eorfreq)'''


 
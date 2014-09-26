# attempt to port some of gsm.f to python.  prm, jan 29, 2013
# the original code is from Angelica de Oliveira-Costa & Max Tegmark 2007
# just trying for minimal '408MHz locked' functionality

# the scheme is to read in the principle compenents from components.dat
# the file is 5 columns. Appears to be frequency, weight1,weight2,weight3 and
# a 'special' weight 4, which is for overall scaling, Angelica says 
# "The "extra" component (ncomp+1) is the overall scaling - we spline its logarithm."

# then read in a healpix binned 3 column ascii file (We'll for sure change 
# this to a fits file soon), which are the three 'component maps' of the '408MHzlocked' model

# next have to interpolate the 'components' meaning the weights I assume,
# then combine and output. seems simple enough (although you wouldn't guess it from 
# the fortran code!)

#import healpy as hp
import pyfits as pf
import scipy.interpolate as sp
from glob import glob
import numpy as np
from pylab import *

#assume we're already pointed at the directory with the data

components=np.loadtxt('components.dat')
#components is 5 values for each of 11 frequencies: freq, w1,w2,w3,w4
#here we make the interpolation functions to be called by freq later do the
#interpolation in loglog space- need to call it with log(freq) as argument


comp1function=sp.UnivariateSpline(np.log(components[:,0]),np.log(components[:,1]),s=0.0001)
comp2function=sp.UnivariateSpline(np.log(components[:,0]),(components[:,2]),s=.00001)
comp3function=sp.UnivariateSpline(np.log(components[:,0]),(components[:,3]),s=.00001,k=2)
comp4function=sp.UnivariateSpline(np.log(components[:,0]),(components[:,4]),s=.00001)



'''f=np.arange(90000)+10
figure(1)
hold(False)
plot(components[:,0],components[:,1],'bx')
hold(True)
plot(f,e**comp1function(log(f)),'red')
xscale('log'),yscale('log')

figure(2)
hold(False)
plot (components[:,0],components[:,2],'bx')
hold(True)
plot (f,comp2function(log(f)),'red')

figure(3)
hold(False)
plot (components[:,0],components[:,3],'bx')
hold(True)
plot (f,comp3function(log(f)),'red')

figure(4)
hold(False)
plot (components[:,0],components[:,4],'bx')
hold(True)
plot (f,comp4function(log(f)),'red')'''


maps408=np.loadtxt('component_maps_408locked.dat')
#will replace with fits next

#function to interpolate weights and make the map
def makemap(freq):
    lfreq=log(freq)
    outmap=e**comp1function(lfreq)*(maps408[:,0]*comp2function(lfreq)+maps408[:,1]*comp3function(lfreq)+maps408[:,2]*comp4function(lfreq))
    return outmap

testmap = makemap(.408)


hdu = pf.PrimaryHDU(testmap)

hdulist = pf.HDUList([hdu])

hdulist.writeto('test_408MHz_4_15_14_2.fits')
hdulist.close()


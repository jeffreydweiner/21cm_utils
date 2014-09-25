
import numpy as np
import pyfits as pyfits
import ephem 
from glob import glob
from matplotlib import mlab
import time
import quaternion as quat
import healpy as hp
import numexpr
import numpy
import scipy.interpolate as sp
import healpy as hp


# Define a local second in Julian time - useful for duration calculation
second_jul = 0.1 / 8640

# Define pi - useful for evaluation expressions
pi = numpy.pi
rtd=180/np.pi
dtr=np.pi/180
e=2.71828

#function to interpolate weights and make the map using gsm galaxy model
def makemap(freq,nside=None):
    lfreq=np.log(freq)
    outmap=np.exp(comp1function(lfreq))*(maps408[:,0]*comp2function(lfreq)+maps408[:,1]*comp3function(lfreq)+maps408[:,2]*comp4function(lfreq))
    if nside:
        outmap=hp.ud_grade(outmap,nside)
    return outmap
    
def cart_to_heal(cartmap,nside):
    """
    routine to bin a cartesian map into a healpix one
    assume cartmap is a rectangular array
    """
    n_b=cartmap.shape[0]
    n_l=cartmap.shape[1]
    thetas=np.pi-np.arange(n_b)*np.pi/n_b
    phis=np.pi-np.arange(n_l)*np.pi*2./n_l
    outmap=np.zeros(hp.nside2npix(nside),dtype=np.float)
    hitmap=np.zeros(hp.nside2npix(nside))
    for i,phi in enumerate(phis):
        for j,theta in enumerate(thetas):
            iring=hp.ang2pix(nside,theta,phi)
            outmap[iring]=outmap[iring]+cartmap[j,i]
            hitmap[iring]=hitmap[iring]+1
    outmap[hitmap>0]=outmap[hitmap>0]/hitmap[hitmap>0]
    return outmap
    
def replace_with_earth(map,elcut=0):
    """
    function assumes input is horizon (az el) coordinate Healpix map. Replace all pixels
    below elcut=0 (in degrees above the horizon) with 300 kelvin. To be made more detailed in some future version
    """
    nside=np.int(np.sqrt(len(map)/12))
    outmap=copy(map)
    pixlist=range(len(map))
    htheta,hphi=hp.pix2ang(nside,pixlist)
    elevation=np.pi/2. -htheta
    outmap[elevation < elcut*dtr]=300.
    return outmap

    
def eqmap2azelmap(map,longi=-104.245,lat=34.4717,ut=12.0,year=2014,month=9,day=15):
    """
    function to rotate celestial coord map to az el with brute force using tools below from Victor Roytman
    """
    julian_date=jdcnv(year,month,day,ut)
    obs=ephem.Observer()
    obs.lon=str(longi)
    obs.lat=str(lat)
    obs.date=ephem.date((year,month,day,ut))
    nside=np.int(np.sqrt(len(map)/12))
    outmap=np.zeros(len(map))
    pixlist=range(len(map))
    htheta,hphi=hp.pix2ang(nside,pixlist)
    elevation=np.pi/2. -htheta
    azimuth=hphi
    ctheta=[]
    cphi=[]
    for az,el in zip(azimuth,elevation):
        ra,dec=obs.radec_of(az,el)
        ctheta.append(np.pi/2. -dec)
        cphi.append(ra)
    #ra,dec=azel2radec(julian_date, azimuth, elevation, lat*dtr, longi)
    #ctheta=np.pi/2.-dec
    #cphi=ra
    ctheta=np.array(ctheta)
    cphi=np.array(cphi)
    rpixlist=hp.ang2pix(nside,ctheta,cphi)
    outmap[pixlist]=map[rpixlist]
    return outmap
    
def galmap2eqmap(map):
    """
    function to rotate galactic coord map (healpix) to equatorial
    """
    nside=np.int(np.sqrt(len(map)/12))
    grot=hp.Rotator(coord='GC')
    pixlist=range(len(map))
    veclist=hp.pix2vec(nside,pixlist)
    rveclist=grot(veclist)
    rpixlist=hp.vec2pix(nside,rveclist[0,:],rveclist[1,:],rveclist[2,:])
    outmap=np.zeros(len(map))
    outmap[rpixlist]=map[pixlist]
    return outmap
    

def jdcnv(year, month, day, hour):
    """Convert Gregorian time (UTC) to Julian Date
    Input:
    year = year (scalar int)
    month = month 1-12 (scalar int)
    day = day 1-31 (scalar int)
    hour = fractional hour (scalar double)
    Output:
    julian = Julian Date (scalar double)
    Original IDL source available at:
    http://idlastro.gsfc.nasa.gov/ftp/pro/astro/jdcnv.pro
    Revision history of IDL source:
    Converted to IDL from Don Yeomans Comet Ephemeris Generator,
    B. Pfarr, STX, 6/15/88
    Converted to IDL V5.0 W. Landsman September 1997
    Added checks on valid month, day ranges W. Landsman July 2008
    """
    year = long(year)
    month = long(month)
    day = long(day)
    # account for leap years
    leap = long((month - 14) / 12.0)
    
    julian = day - 32075L + long((1461L) * (year + 4800L + leap) / 4.0) \
             + long((367L) * (month - 2 - leap * 12) / 12.0) \
             - long(3 * (long((year + 4900L + leap) / 100.0)) / 4.0) \
             + (hour / 24.0) - 0.5
    
    return julian

def ct2lst(julian_date):
    """Convert Civil Time (as Julian date) to Greenwich Sidereal Time
    Input:
    julian_date = Julian date (scalar or vector double)
    Output:
    gst = Greenwich Sidereal Time (scalar or vector double)
    The constants used in ct2lst come from Astronomical Algorithms by Jean
    Meeus, p. 84 (Eq. 11-4).
    Original IDL source available at:
    http://idlastro.gsfc.nasa.gov/ftp/pro/astro/ct2lst.pro
    Revision history of IDL source:
    Adapted from the FORTRAN program GETSD by Michael R. Greason, STX,
    27 October 1988.
    Use IAU 1984 constants Wayne Landsman, HSTX, April 1995, results
    differ by about 0.1 seconds
    Longitudes measured *east* of Greenwich W. Landsman December 1998
    Time zone now measure positive East of Greenwich W. Landsman July 2008
    Remove debugging print statement W. Landsman April 2009
    """
    c1 = 280.46061837
    c2 = 360.98564736629
    c3 = 0.000387933
    c4 = 38710000.0
    t0 = numexpr.evaluate('julian_date - 2451545.0')
    t = numexpr.evaluate('t0 / 36525')
    
    # Compute GST in seconds
    theta = numexpr.evaluate('c1 + (c2 * t0) + (t**2)'\
                             ' * (c3 - (t / c4))')
    
    # Compute LST in hours
    gst = numexpr.evaluate('theta / 15.0')
    
    # Deal with LST out of bounds
    negative = numexpr.evaluate('gst < 0.0')
    negative_gst = gst[negative]
    negative_gst = numexpr.evaluate('24.0 + negative_gst % 24')
    gst[negative] = negative_gst
    gst = numexpr.evaluate('gst % 24.0')
    
    return gst

    # IMPORTANT: Latitude is in RADIANS
    # Longitude is in DEGREES
    # in order to reduce function calls
def azel2radec(julian_date, azimuth, elevation, latitude, longitude_deg):
    """Convert from horizon coordinates (azimuth-elevation) to celestial
    coordinates (right ascension-declination)
    Input:
    julian_date = Julian date (scalar or vector double)
    azimuth = azimuth in radians (scalar or vector double)
    elevation = elevation in radians (scalar or vector double)
    latitude = latitude in radians north of the equator (scalar or
    vector double)
    longitude_deg = longitude in degrees east of the prime meridian
    (scalar or vector double)
    Output:
    ra = right ascension in radians (scalar or vector double)
    dec = declination in radians (scalar or vector double)
    Original IDL source available at:
    http://cosmology.berkeley.edu/group/cmbanalysis/forecast/idl/azel2radec.pro
    Revision history of IDL source:
    Created, Amedeo Balbi, August 1998 (based partly on material
    by Pedro Gil Ferreira)
    Modified, Amedeo Balbi, October 1998, to accept vectors as input
    """
    
    # Rescale azimuth
    azimuth = numexpr.evaluate('azimuth - pi')
    
    # Get the Greenwich Sidereal Time of the date
    gst = ct2lst(julian_date)
    
    # Get the Local Sidereal Time
    lst = numexpr.evaluate('gst + longitude_deg / 15.0')
    
    # Calculate the declination
    dec = numexpr.evaluate('arcsin( '\
                           'sin(elevation) * sin(latitude) - '\
                           'cos(elevation) * cos(azimuth) * cos(latitude)'\
                           ' )')
    
    # Calculate the right ascension from the hour angle
    ha = numexpr.evaluate('arctan2( '\
                          '-(cos(elevation)*sin(azimuth)) , '\
                          '(sin(elevation)*cos(latitude)-'\
                          'cos(elevation)*cos(azimuth)*sin(latitude))'\
                          ' )')
    ha = numexpr.evaluate('24 * ha / 2.0 / pi')
    ra = numexpr.evaluate('lst - ha')
    ra=lst*2*pi/24. - ha
    #Deal with RA out of bounds
    negative = numexpr.evaluate('ra<0.0')
    negative_ra = ra[negative]
    negative_ra = numexpr.evaluate('24.0 + negative_ra % 24')
    ra[negative] = negative_ra
    ra = numexpr.evaluate('ra % 24.0')
    
    # Convert from hours to radians
    ra = numexpr.evaluate('ra * pi / 12')
    
    return ra, dec
    
# IPython log file

import quaternion as quat
get_ipython().magic(u'cd /smbmount/users/peterm/software/zonca_python_tools/')
import quaternion as quat
get_ipython().magic(u'cd /smbmount/labuse/EXPT/21\\ cm/Galactic\\ Emission/gsm')
get_ipython().magic(u'paste')
import healpy as hp
get_ipython().magic(u'paste')
out45=makemap(45)
get_ipython().magic(u'paste')
get_ipython().magic(u'pwd ')
get_ipython().system(u'ls -F --color ')
get_ipython().magic(u'logstart')
manymaps=[]
for freq in range(50,500,5):
    manymaps.append(makemap(freq))
    
manymaps32=[]
for map in manymaps:
    manymaps32.append(hp.ud_grade(map,32))
    
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/2.0)
    
hp.mollview manymaps32[40]
get_ipython().magic(u'autocall 2')
hp.mollview(manymaps32[40])
jnk= manymaps32[40]+100.
hp.mollview(jnk)
get_ipython().magic(u'logstart')
get_ipython().magic(u'paste')
lowfreq=50
hifreq=500
stepfreq=5
ri_start=150
ri_stop=170

speclen=(hifreq-lowfreq)/stepfreq
eorsig=zeros(speclen)
eorfreq=arange(lowfreq,hifreq,stepfreq)
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

manymaps=[]
for freq in range(lowfreq,hifreq,stepfreq):
    manymaps.append(makemap(freq))
manymaps32=[]
for i,map in enumerate(manymaps):
    map=map+eorsig[i]
    manymaps32.append(hp.ud_grade(map,32))
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/2.0)
hp.mollview(manymaps32[40])
eorfreq(40)
eorfreq[40]
eorfreq[10]
eorfreq[60]
eorfreq[30]
eorfreq[20]
hp.mollview(manymaps32[20])
hp.mollview(manymaps32[21])
hp.mollview(manymaps32[22])
hp.mollview(manymaps32[23])
eorsig[23]
eorsig[20]
specaz0l=log10(specaz0)
spec45=zeros([90,360])
ellist=arange(100)*np.pi/100.-np.pi/2
azlist=arange(100)*2*np.pi/100.
pixaz60=zeros(120,dtype=np.integer)-60.*dtr
pix45=zeros(360,dtype=np.integer)
for i,el in enumerate(arange(-60,60)):
    pixaz60[i]=hp.ang2pix(32,np.pi/2. - el*dtr,-60.*dtr)
    
spec=zeros([90,360])
for j in range(90):
    for i,pix in enumerate(pixaz60):
        spec[j,i]=manymaps32[j][pix]
        
figure()
pcolormesh(spec)
spec.shape
hold(False)
plot(spec[:,0])
plot(eorfreq,spec[:,0])
get_ipython().magic(u'A')
get_ipython().magic(u'paste')
ri_stop=155

speclen=(hifreq-lowfreq)/stepfreq
eorsig=zeros(speclen)
eorfreq=arange(lowfreq,hifreq,stepfreq)
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

manymaps=[]
for freq in range(lowfreq,hifreq,stepfreq):
    manymaps.append(makemap(freq))
manymaps32=[]
for i,map in enumerate(manymaps):
    map=map+eorsig[i]
    manymaps32.append(hp.ud_grade(map,32))
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/2.0)


pixaz60=zeros(120,dtype=np.integer)-60.*dtr
for i,el in enumerate(arange(-60,60)):
    pixaz60[i]=hp.ang2pix(32,np.pi/2. - el*dtr,-60.*dtr)

spec=zeros([90,360])
for j in range(90):
    for i,pix in enumerate(pixaz60):
        spec[j,i]=manymaps32[j][pix]
hold (True)
plot(eorfreq,spec[:,0])
get_ipython().magic(u'paste')
stepfreq=5
ri_start=150
ri_stop=150

speclen=(hifreq-lowfreq)/stepfreq
eorsig=zeros(speclen)
eorfreq=arange(lowfreq,hifreq,stepfreq)
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

manymaps=[]
for freq in range(lowfreq,hifreq,stepfreq):
    manymaps.append(makemap(freq))
manymaps32=[]
for i,map in enumerate(manymaps):
    map=map+eorsig[i]
    manymaps32.append(hp.ud_grade(map,32))
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/2.0)


pixaz60=zeros(120,dtype=np.integer)-60.*dtr
for i,el in enumerate(arange(-60,60)):
    pixaz60[i]=hp.ang2pix(32,np.pi/2. - el*dtr,-60.*dtr)

spec=zeros([90,360])
for j in range(90):
    for i,pix in enumerate(pixaz60):
        spec[j,i]=manymaps32[j][pix]
figure()
pcolormesh(spec)
plot(eorfreq,eorsig)
plot(eorfreq,spec[:,0],'+')
plot(eorfreq,spec[:,0],'+')
get_ipython().magic(u'paste')
lowfreq=50
hifreq=500
stepfreq=5
ri_start=150
ri_stop=150

speclen=(hifreq-lowfreq)/stepfreq
eorsig=zeros(speclen)
eorfreq=arange(lowfreq,hifreq,stepfreq)
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

manymaps=[]
for freq in range(lowfreq,hifreq,stepfreq):
    manymaps.append(makemap(freq))
manymaps32=[]
for i,map in enumerate(manymaps):
    map=map+eorsig[i]
    manymaps32.append(hp.ud_grade(map,32))
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/4.0)


pixaz60=zeros(120,dtype=np.integer)
for i,el in enumerate(arange(-60,60)):
    pixaz60[i]=hp.ang2pix(32,np.pi/2. - el*dtr,-60.*dtr)

spec=zeros([90,120])
for j in range(90):
    for i,pix in enumerate(pixaz60):
        spec[j,i]=manymaps32[j][pix]
plot(eorfreq,spec[:,0])
figure()
pcolormesh(spec)
plot(eorfreq,spec[:,119])
figure(5)
plot(eorfreq,spec[:,119])
figure(5)
plot(eorfreq,spec[:,119])
grid()
(170-130)/.01
get_ipython().magic(u'paste')
lowfreq=130
hifreq=170
stepfreq=.01
ri_start=150
ri_stop=150

speclen=(hifreq-lowfreq)/stepfreq
eorsig=zeros(speclen)
eorfreq=arange(lowfreq,hifreq,stepfreq)
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

manymaps=[]
for freq in range(lowfreq,hifreq,stepfreq):
    manymaps.append(makemap(freq))
manymaps32=[]
for i,map in enumerate(manymaps):
    map=map+eorsig[i]
    manymaps32.append(hp.ud_grade(map,32))
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/4.0)


pixaz60=zeros(120,dtype=np.integer)
for i,el in enumerate(arange(-60,60)):
    pixaz60[i]=hp.ang2pix(32,np.pi/2. - el*dtr,-60.*dtr)

spec=zeros([90,120])
for j in range(90):
    for i,pix in enumerate(pixaz60):
        spec[j,i]=manymaps32[j][pix]
get_ipython().magic(u'paste')
maps408=loadtxt('component_maps_408locked.dat')

manymaps=[]
for freq in eorfreq:
    manymaps.append(makemap(freq))
manymaps32=[]
for i,map in enumerate(manymaps):
    map=map+eorsig[i]
    manymaps32.append(hp.ud_grade(map,32))
for i,map in enumerate(manymaps32):
    map=galmap2eqmap(map)
    map=eqmap2azelmap(map,ut=0.0)
    map=replace_with_earth(map)
    manymaps32[i]=hp.smoothing(map,np.pi/4.0)


pixaz60=zeros(120,dtype=np.integer)
for i,el in enumerate(arange(-60,60)):
    pixaz60[i]=hp.ang2pix(32,np.pi/2. - el*dtr,-60.*dtr)

spec=zeros([90,120])
for j in range(90):
    for i,pix in enumerate(pixaz60):
        spec[j,i]=manymaps32[j][pix]

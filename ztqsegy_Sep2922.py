import sys, os.path
import argparse
import numpy as np
from math import sqrt, fabs
from scipy.spatial import cKDTree as KDTree
import pandas as pd
import segyio
from shutil import copyfile
from scipy import interpolate
from datetime import datetime



"""
Depth conversion using quadratic equation

5/Mar/2012 11:25 : created zqcon.py
10/Mar/2012 20:49 : added time conversion

+Sample input file: k1tx3.lst
   503872.34   2925323.47       1602
   504870.21   2924325.58       1593
   505868.09   2925323.47       1582
   506865.96   2928317.12       1577
   506865.96   2925323.47       1575
   506865.96   2922329.81       1592
   507863.83   2927319.24       1571
   507863.83   2924325.58       1567

+Sample coefficient file: qcoef.dat
   571199.80    2919956.00     -43.385    7719.345    6801.382       0.000    7520.170    7005.920            NF-4A      0.026       0.754         131        9665
   567835.30    2937475.50     -96.197    8094.850    6135.446       0.000    7663.217    6568.652             NF-5      0.026       0.771         130        9819
   594051.20    2834248.00     191.077    7404.620    7306.613       0.000    8243.639    6484.981           NF-6AA      0.017       0.791          87       10688
   602468.00    2928893.00    -269.791    7509.786    7394.564       0.000    6346.340    8546.043            NFB-1      0.035       0.755         179        9576
   607200.90    2926207.00      77.392    6634.064    7452.582       0.000    6966.907    7126.673           NFB-19      0.036       0.774         181        9710
   607206.70    2926206.10     -61.118    6949.164    7326.107       0.000    6700.548    7560.820           NFB-25      0.036       0.777         182        9752
   631608.00    2893358.00     109.695    6781.596    7002.654       0.000    7218.535    6606.951           NFD-1B      0.034       0.838         173       10759
this file was generated from wvel.py

>python qzcon0.py -fk1tx3.lst  -q 0 1 2 3 4 -eqcoef.dat

+ To time convert a depth file with spatially varying coef
Sample input file : k1tz.dat
   537800.00    2902372.09       0.667        8449        12668         155        7407        7539
   537800.00    2899378.44       0.665        8492        12760         191        7170        7970
   537800.00    2896384.78       0.658        8440        12816         222        6953        8391
   538797.87    2964241.01       0.779        9558        12269         -92        7454        6332
   538797.87    2961247.36       0.765        9351        12215         -82        7693        6047
   538797.87    2958253.70       0.746        9021        12101         -85        7862        5838
   538797.87    2955260.04       0.743        8988        12096         -86        7980        5696
   538797.87    2952266.38       0.741        8957        12096         -86        8048        5623
   538797.87    2949272.73       0.728        8768        12044         -85        8076        5610
   538797.87    2946279.07       0.732        8868        12115         -73        8170        5525
   538797.87    2943285.41       0.721        8747        12141         -65        8194        5604


C:\sekpy\velocity>python qzcon0.py -fk1tz.dat -d 0 1 3 -t -eqcoef.dat -q 0 1 2 3 4

+ To depth convert using a single function:
C:\sekpy\velocity>python qzcon0.py -fk1tz.dat -d 0 1 3  -c 38 7052 6994


+ To time convert using a single function:
C:\sekpy\velocity>python qzcon0.py -fk1tz.dat -d 0 1 3 -t -c 38 7052 6994

+using monte carlo simulation us -S standard deviation for 3 coefs and percentile to
    compute at -P 0.15 and resample grid every 3rd value. Input data is in Surfer grid format
C:\\Users\20504\sami\SEKDATA\wdata>python ztqmc.py -f 10REG_SUDSH_400X_grid.grd
-i surfer -c 0.000    7591.537    6573.168 -S 0.0 650 700 -T -2.0 2.0  -P 0.15
-G 3


+input is zmap grid sampled every 2nd point
C:\\Users\20504\sami\SEKDATA\wdata>python ztqmc.py -f 10REG_SUDSH_400X_grid.dat -
i zmap -c 0.000    7591.537    6573.168 -S 0.0 650 700 -T -2.0 2.0  -P 0.90 -G 2
 >10REG_SUDSH_400X_grid_p90x2.lst


+ztqmc.py has changed after that: triggering monte carlo by using -R

+ depth conversion using monte carlo of a surfer grid
C:\\Users\Administrator\sekdata\velocities>python ztqmc.py -fsdtba.grd -i surfer
-c 0.000    7591.537    6573.168 -R -S 0.0 650 700 -T -2.0 2.0  -P 0.85 > t

+time conversion using monte carlo. -F plotting of histograms does not show?? resulting
    data seems ok.
C:\\Users\Administrator\sekdata\velocities>python ztqmc.py -f t -i xyz -c 0.000
  7591.537    6573.168 -R -S 0.0 650 700 -T -2.0 2.0  -P 0.85 -t -F

+straight time conversion
C:\\Users\Administrator\sekdata\velocities>python ztqmc.py -f t -i xyz -c 0.000
  7591.537    6573.168 -t

*** In depth conversion the input time is assumed t2way. The output is listed as t1w

*** In time conversion the output is listed as t1way
10/Mar/2012 21:06 done editing saved as qzcon.py
11/Mar/2012 8:45: changed name to ztq.py
28/Feb/2013 8:08 added t2w in ms to listing of both time and depth conversion
    modified listing to be x y z t1w t2w vav ....
19/Mar/2013 13:24  added onse space before listing in print to match to wvel.py output
28/Apr/2013 8:10 added space to printing to mathch wvel.py listing
28/Jan/2015 added surfer grid input and probability monte carlo
    modified all functions to print internally and not return data
    added input coef files from surfer grid
29/Jan/2015: changed a lot of command line options. Added input zgrid for both data
    and coef files.
01 Feb 2015: added printing to stderr every 100 point. Changed # of mcruns to 250 only
    added plotting of histogram and fitted pdf for every point. To exit use ctrl C
09 Feb 2015: added -R to switch on Monte Carlo simulation
    added discrete distribution of simulated zmc
10 feb 2015: fitted a continous norm distribution to mc'd z and t
    found the pctl from the norm distribution.
    Plotting of depth conversion is OK but in time conversion is a problem?
    data seems to convert both ways OK

*depth to time conversion with 1 qcoefs
python ztqmc.py time1300.txt --datahdrlines 1 --timeconvert --datafiletype xyz --datacols 1 2 4 --qcoefs 0 4750 3400
*time to depth conversion with 1 qcoefs
python ztqmc.py time1300.txt --datahdrlines 1 --datafiletype xyz --datacols 1 2 3 --qcoefs 0 4750 3400

*time to depth conversion with quadratic coeffs file
python ztqmc.py time1300.txt --datahdrlines 1  --datafiletype xyz --datacols 1 2 4 --qcfilename time1300.txt --qchdrlines 1 --qccols 1 2 6 7 8 --timeconvert

*depth convert to time using quadratic coeffs file
python ztqmc.py time1300.txt --datahdrlines 1  --datafiletype xyz --datacols 1 2 3 --qcfilename time1300.txt --qchdrlines 1 --qccols 1 2 6 7 8

python ztq.py time1300.txt --datacols 1 2 3 --datahdrlines 1 --qcfilename time1300.txt --qccols 1 2 6 7 8

python ztqsegy.py aix.sgy --datafiletype segy --qcoefs 0 4740 3440

python ztqsegy.py aix.sgy --datafiletype segy --qcfilename time1300_tcqf.txt --qccols 0 1 6 7 8 --outrange 0 12000
python ztqsegy.py UL30.dat --datafiletype xyz --datacols 0 1 2 --datahdrlines 20 --qcfilename P7300_999_FINAL_STACKING_VEL_zvr_qcf.txt --qchdrlines 1 --qccols 1 2 6 7 8

"""

def quadeval(cf,xi):
    """
    if doprob:
        zmc= np.zeros(shape=mcnum)
        zx= np.arange(mcnum)
        for i in range(mcnum):
            zmc[i]=mc_quadeval_draw(cf,xi)

        zmcsorted = np.sort(zmc)
        zmcmean= np.mean(zmc)
        zmcstd= np.std(zmc)
        zmcprob= st.norm(zmcmean,zmcstd)
        z= zmcprob.ppf(pctl)
        if pdfplot:
            plt.hist(zmcsorted,normed=1)
            fit = st.norm.pdf(zmcsorted,zmcmean,zmcstd)
            plt.plot(zmcsorted,fit,'r')
            plt.show()
    else:
    """
    z=cf[0]+ cf[1]*xi + cf[2] * xi * xi
    return z


"""
def mc_quadroots_draw(cf0,cf1,cf2,xi):
    a0=st.norm(cf0,coefstd[0]).rvs()   #a0d for distribution
    a1=st.norm(cf1,coefstd[1]).rvs()
    a2=st.norm(cf2,coefstd[2]).rvs()
    xi += st.uniform(t2wmc[0],t2wmc[1]).rvs()
    a0 =cf0- xi
    a1 = cf1
    a2 =cf2

    num=(a1 *a1) - 4.0 * a2 * a0
    if num >= 0.0:
        r1=(-a1 + sqrt(fabs(num)))/(2.0 * a2)
        r2 = (-a1 - sqrt(fabs(num)))/(2.0 * a2)
    else:
        r1= -a1/ (2.0 *a2)
        r2= sqrt(fabs(num))/(2.0 *a2)
    return r1,r2
"""


def quadroots(cf0,cf1,cf2,xi):
    """
    if doprob:
        tmc1= np.zeros(shape=mcnum)
        tmc2= np.zeros(shape=mcnum)
        tx= np.arange(mcnum)
        for i in range(mcnum):
            tmc1[i],tmc2[i]=mc_quadroots_draw(cf0,cf1,cf2,xi)


        tmc1sorted = np.sort(tmc1)
        tmc1mean = np.mean(tmc1)
        tmc1std = np.std(tmc1)
        tmc1prob = st.norm(tmc1mean,tmc1std)
        r1 = tmc1prob.ppf(pctl)

        tmc2sorted = np.sort(tmc2)
        tmc2mean = np.mean(tmc2)
        tmc2std = np.std(tmc2)
        tmc2prob = st.norm(tmc2mean,tmc2std)
        r2 = tmc2prob.ppf(pctl)

        #only plot first root
        if pdfplot:
            plt.hist(tmc1sorted,normed=1)
            fit = st.norm.pdf(tmc1sorted,tmc1mean,tmc1std )
            plt.plot(tmc1sorted,fit,'r')
            plt.show()
    else:
    """
    a0 =cf0- xi
    a1 = cf1
    a2 =cf2


    num=(a1 *a1) - 4.0 * a2 * a0
    if num >= 0.0:
        r1=(-a1 + sqrt(fabs(num)))/(2.0 * a2)
        r2 = (-a1 - sqrt(fabs(num)))/(2.0 * a2)
    else:
        r1= -a1/ (2.0 *a2)
        r2= sqrt(fabs(num))/(2.0 *a2)
    return r1,r2




#*****************qhull code
link = lambda a,b: np.concatenate((a,b[1:]))
edge = lambda a,b: np.concatenate(([a],[b]))
def qhull(sample):
    def dome(sample,base):
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, 0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
    	axis = sample[:,0]
    	base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
    	return link(dome(sample, base),dome(sample, base[::-1]))
    else:
        return sample


#************** end qhull code


def pip(x,y,poly):  #point_in_poly

   # check if point is a vertex
   if (x,y) in poly: return True

   # check if point is on a boundary
   for i in range(len(poly)):
      p1 = None
      p2 = None
      if i==0:
         p1 = poly[0]
         p2 = poly[1]
      else:
         p1 = poly[i-1]
         p2 = poly[i]
      if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
         return True

   n = len(poly)
   inside = False

   p1x,p1y = poly[0]
   for i in range(n+1):
      p2x,p2y = poly[i % n]
      if y > min(p1y,p2y):
         if y <= max(p1y,p2y):
            if x <= max(p1x,p2x):
               if p1y != p2y:
                  xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
               if p1x == p2x or x <= xints:
                  inside = not inside
      p1x,p1y = p2x,p2y

   if inside: return True
   else: return False



#...............................................................................
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]

def idw(xy,vr,xyi):

#    xyt,vr=datain('aqv3stk2.lst')
#    print "size of xyt %10d , size of vr %10d" % (xyt[:,0].size,vr.size)
#   xyti=data2in('k1tx3.lst')
    # N = vr.size
    # # Ndim = 2
    # Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 2  # weights ~ 1 / distance**p

    invdisttree = Invdisttree( xy, vr, leafsize=leafsize, stat=1 )
    interpol = invdisttree( xyi, nnear=Nnear, eps=eps, p=p )
    return interpol


def surfergrid2xyz(fname,nullvalue,mltp,nskip):
    z=[]
    xy=[]
    fgrid=open(fname,'r')
    # headercode= fgrid.readline()
    header1=fgrid.readline()
    xnodes,ynodes=list(map(int,header1.split()))
    header2=fgrid.readline()
    xmin,xmax=list(map(float,header2.split()))
    header3=fgrid.readline()
    ymin,ymax=list(map(float,header3.split()))
    header4=fgrid.readline()
    zmin,zmax=list(map(float,header4.split()))
    xinc=(xmax-xmin)/(xnodes-1)
    yinc=(ymax -ymin)/(ynodes-1)
#    print xnodes, ynodes, xmin, xmax, ymin,ymax, zmin, zmax
#   print xinc, yinc

    # ycounter=0
    # xcounter=0
    for j in range(ynodes):
        yc=ymin +(yinc *j)
        for k in range(xnodes):
            oneline=fgrid.readline()
            flds=oneline.split()
            nflds=len(flds)
            if nflds >0:
                for m0 in range(nflds):
                    if flds[m0] != nullvalue:
                        xc = xmin + (xinc  * k)
                        zc=float(flds[m0])
                        z.append(zc)
                        xy.append((xc,yc))
                    k += 1
#                    print "j: %5d  k:%5d  xc:%12.2f  yc: %12.2f  z:%10s"\
#                    %(j,k,xc,yc,flds[m])

    fgrid.close()
    xyarray=np.array(xy[::nskip][:])
    zarray=np.array(z[::nskip])
    zarray *=mltp
    return xyarray,zarray,xnodes,ynodes,xmin,xmax,ymin,ymax





def zmapgridin(fname,mltp,nskip):
    fgrid=open(fname,'r')
    header=True
    firstat=True
    while header==True:
        line=fgrid.readline()
#        print(line[:-1])
        flds=line.split()
        if (flds[0][0]=='@' and firstat):
            firstat=False
            line1=fgrid.readline()
            line1flds=line1.split(',')
            line2=fgrid.readline()
            line2flds=line2.split(',')
            nullval=line1flds[1].strip()
            nrows=int(line2flds[0])
            ncols=int(line2flds[1])
            minx=float(line2flds[2])
            maxx=float(line2flds[3])
            miny=float(line2flds[4])
            maxy=float(line2flds[5])
            dx=(maxx-minx)/ncols
            dy=(maxy-miny)/nrows
#            print( dx,dy)
        elif (flds[0][0]=='@' and firstat==False):#add check for + in zmap grid
                                                #just delete line with + from grid
            header=False
    xstart=minx
    ystart=maxy

    zval=[]
    nlines=0
#    print ("Null value %s ", nullval)
    for line in fgrid:
#        if line[0][0] != ' ':
#            continue
#9/Jul/2012 13:06 removed it and prog works OK
        lineflds=line.split()
        nlines= nlines+1
#        print(nlines,lineflds)


        for i in range(len(lineflds)):
            zval.append(lineflds[i])
    xy=[]
    z=[]
    k=-1
    for i in range(ncols):
        for j in range(nrows):
            k=k+1
#            print "k: %5d" % k
            if str(zval[k])== nullval:
                pass
#                print(zval[i+j],nullval)
            else:
                xc= xstart + i * dx
                yc = ystart - j * dy
#                print("%10.2f %10.2f %10.0f " % (float(xc),float(yc), float(zval[k])))
                xy.append((xc,yc))
                z.append(float(zval[k]))
#    print("len of xyz:",len(xyz))
    fgrid.close()
    #xyarray=np.array(xy[::nskip][::nskip])
    xyarray=np.array(xy[::nskip][:])
    zarray=np.array(z[::nskip])
    zarray *=mltp
    return xyarray,zarray,ncols,nrows,minx,maxx,miny,maxy




def datain(fname,cols,mltp,skiphdr=0,nskip=1):
    xy=np.genfromtxt(fname,usecols=(cols[0],cols[1]),skip_header=skiphdr)
    t=np.genfromtxt(fname,usecols=(cols[2]),skip_header=skiphdr)
    t  *=mltp

    xyarray=np.array(xy[::nskip][:])
    tarray=np.array(t[::nskip])


    #return xy,t
    return xyarray,tarray




def qcoefin(fname,qccols,skiphdr=0,nskip=1):
    xy=np.genfromtxt(fname,usecols=(qccols[0],qccols[1]),skip_header=skiphdr)
    qc=np.genfromtxt(fname,usecols=(qccols[2],qccols[3],qccols[4]),skip_header=skiphdr)
    xyarray=np.array(xy[::nskip][:])
    qcarray=np.array(qc[::nskip])
    return xyarray,qcarray

def acoefin(fname,xc,yc,ac,skiphdr=0,nskip=1):
    xy=np.genfromtxt(fname,usecols=(xc,yc),skip_header=skiphdr)
    a=np.genfromtxt(fname,usecols=(ac),skip_header=skiphdr)
    xyarray=np.array(xy[::nskip][:])
    aarray=np.array(a[::nskip])
    return xyarray,aarray




def listdatain(xy,t):
    for i in range(t.size):
        print("%10.2f  %10.2f  %10.0f " %(xy[i,0],xy[i,1],t[i]))

def listqc(xyqc,qcoef):
    for i in range(xyqc[:,0].size):
        print("%12.2f  %12.2f  %10.2f  %10.2f  %10.2f " % \
        (xyqc[i,0],xyqc[i,1],qcoef[i,0],qcoef[i,1],qcoef[i,2]))


def listzconv(xy,t,z,a0,a1,a2):
    t1w= t/ 2000.0
    vav= z/t1w
    for i in range(z.size):
        print(" %12.2f  %12.2f  %10.0f  %10.3f  %10.0f   %10.0f  %10.0f  %10.0f  %10.0f " %\
        (xy[i,0],xy[i,1],z[i],t1w[i],t[i],vav[i],a0[i],a1[i],a2[i]))


def listtconv(xy,t1w,z,a0,a1,a2):

    vav= z/t1w
    t2w= t1w *2000.0
    for i in range(z.size):
        print(" %12.2f  %12.2f  %10.0f  %10.3f   %10.0f  %10.0f  %10.0f  %10.0f " %\
        (xy[i,0],xy[i,1],z[i],t1w[i],t2w[i],vav[i],a0[i],a1[i],a2[i]))





def datain_kngdmflt(fname,mltp):
    kdf = pd.read_csv(fname,delim_whitespace=True,header = None)
    kdf.head()
    xc = kdf.iloc[:,2].values
    print('in data in',xc.min())
    yc = kdf.iloc[:,3].values
    print(yc.max())
    t2w = kdf.iloc[:,4].values
    segid = kdf.iloc[:,6].values
    fltname = kdf.iloc[:,5]
    print(xc.size,segid.min())


    xy = np.vstack((xc,yc)).T
    return xy,t2w,segid,fltname

def list_kngdmflt(fltfilename,xy,tc,segid,fltname):
    with open(fltfilename,'w') as fp:
        for  i in range(tc.size):
           print('unspecified     0.0 {:.1f} {:.1f} {:.2f} {:} {:}   unspecified'
                    .format(xy[i,0],xy[i,1],tc[i],fltname[i],segid[i]),file=fp)





def zconv1xyz(xy,t,qc,kflt=False):
    t1w = t /2000.0
    zc = np.array(t)
    vav = np.array(t)
    if not kflt:
        zc[0] = t1w[0] = vav[0] = 0
        nstart = 1
    else:
        nstart = 0
    for i in range(nstart,t.size):
        zc[i]=quadeval(qc,t1w[i])
        vav[i] = zc[i]/t1w[i]
    return zc, vav




def tconv1xyz(xy,z,qc,kflt=False):
    vav = np.array(z)
    t0 = np.array(z)
    zc = np.array(z)
    if not kflt:
        zc[0] = t0[0] = vav[0] = 0
        nstart = 1
    else:
        nstart = 0
    for i in range(nstart,z.size):
        t0[i],t1=quadroots(qc[0],qc[1],qc[2],z[i])
        vav[i]=z[i]/t0[i]
        # t2w= t0[i] * 2000.0
    return t0, vav


def segyzconvfile(xy,t,xyqc,qcoef):
    t1w =t/2000.0
    a0=idw(xyqc,qcoef[:,0],xy)
    a1=idw(xyqc,qcoef[:,1],xy)
    a2=idw(xyqc,qcoef[:,2],xy)
    zc=np.array(t)
    vav = np.array(t)
    zc[0] = t1w[0] = vav[0] = 0
    for i in range(1,t.size):
        zc[i]=quadeval([a0,a1,a2],t1w[i])
        vav[i] = zc[i]/t1w[i]

    return zc, vav, a0, a1, a2



def segytconvfile(xy,z,xyqc,qcoef):
    a0=idw(xyqc,qcoef[:,0],xy)
    a1=idw(xyqc,qcoef[:,1],xy)
    a2=idw(xyqc,qcoef[:,2],xy)
    t1w = np.array(z)
    vav = np.array(z)
    zc = np.array(z)
    zc[0] = t1w[0] = vav[0] = 0
    for i in range(1,z.size):
        t1w[i],t1=quadroots(a0,a1,a2,z[i])
        vav[i]=z[i]/t1w[i]
        # t2w= t0[i] * 2000.0

    return t1w,vav,a0,a1,a2






def zconvfile(xy,t,xyqc,qcoef):
    t1w =t/2000.0
    a0list=idw(xyqc,qcoef[:,0],xy)
    a1list=idw(xyqc,qcoef[:,1],xy)
    a2list=idw(xyqc,qcoef[:,2],xy)
    zc=np.array(t)
    vav = np.array(t)

    for i in range(xy[:,0].size):
        zc[i]=quadeval([a0list[i],a1list[i],a2list[i]],t1w[i])
        vav[i]=zc[i]/t1w[i]

    return zc, vav, a0list, a1list, a2list

def tconvfile(xy,z,xyqc,qcoef):
    a0list=idw(xyqc,qcoef[:,0],xy)
    a1list=idw(xyqc,qcoef[:,1],xy)
    a2list=idw(xyqc,qcoef[:,2],xy)
    t1w = np.array(z)
    vav = np.array(z)
    for i in range(xy[:,0].size):
        t0,t1=quadroots(a0list[i],a1list[i],a2list[i],z[i])
        t1w[i]=t0
        vav[i]=z[i]/t1w[i]

    return t1w,vav,a0list,a1list,a2list


def segyzconvqfile(xy,t,xy0,a0,xy1,a1,xy2,a2):
    t1w =t/2000.0
    a0=idw(xy0,a0,xy)
    a1=idw(xy1,a1,xy)
    a2=idw(xy2,a2,xy)
    zc = np.array(t)
    vav = np.array(t)
    zc[0] = t1w[0] = vav[0] = 0
    for i in range(1,t.size):
        zc[i]=quadeval([a0,a1,a2],t1w[i])
        vav[i] = zc[i]/t1w[i]

    return zc,vav,a0,a1,a2


def segytconvqfile(xy,z,xy0,a0,xy1,a1,xy2,a2):
    a0=idw(xy0,a0,xy)
    a1=idw(xy1,a1,xy)
    a2=idw(xy2,a2,xy)
    t1w=np.array(z)
    vav = np.array(z)
    zc = np.array(z)
    t0 = np.array(z)
    zc[0] = t0[0] = vav[0] = 0
    for i in range(1,z.size):
        t1w[i],t1=quadroots(a0,a1,a2,z[i])
        vav[i]=z[i]/t1w[i]
        # t2w= t1w[i] * 2000.0


    return t1w,vav,a0,a1,a2






def zconvqfile(xy,t,xy0,a0,xy1,a1,xy2,a2):
    t1w =t/2000.0
    a0list=idw(xy0,a0,xy)
    a1list=idw(xy1,a1,xy)
    a2list=idw(xy2,a2,xy)
    zc = np.array(t)
    vav = np.array(t)
    for i in range(xy[:,0].size):
        zc[i]=quadeval([a0list[i],a1list[i],a2list[i]],t1w[i])
        vav[i]=zc[i]/t1w[i]

    return zc,vav,a0list,a1list,a2list


def tconvqfile(xy,z,xy0,a0,xy1,a1,xy2,a2):
    a0list=idw(xy0,a0,xy)
    a1list=idw(xy1,a1,xy)
    a2list=idw(xy2,a2,xy)
    t1w=np.array(z)
    vav = np.array(z)
    for i in range(xy[:,0].size):# len of xy data needs correction???
        t1w[i],t1=quadroots(a0list[i],a1list[i],a2list[i],z[i])
        vav[i]=z[i]/t1w[i]
    return t1w,vav,a0list,a1list,a2list



def getcommandline():
    parser= argparse.ArgumentParser(description='Depth Conversion using quadratic coefficients ')
    parser.add_argument('datafilename',help='Horizon Data file to be depth or time converted')

    parser.add_argument('--datafiletype',choices=['xyz','zmap','surfer','segy','kngdmflt'],default='xyz',
            help='Input file type: flat ASCII, zmap grid, surfer grid, segy, kingdom fault. dfv=xyz')

    """
    cannot specify output format till I check if the input was a grid. flat xyz might not fit
    a grid output format. Problem with handling null values because I remove them to compute.
    parser.add_argument('-i','--datafiletype',choices=['xyz','zmap','zmap2surfer','surfer','surfer2zmap'],default='xyz',\
    help='Output file type: flat ASCII, zmap in/out,zmap in/surfer out,surfer in/out,surfer in zmapout. dfv=flat ASCII')
    """
    parser.add_argument('--datacols',type=int,nargs=3,default=[0,1,2],
            help=' Use if xyz flat file: xcol ycol tcol  3 values expected. dfv=0 1 2')
    parser.add_argument('--nullvalue',default='1.70141e+038',
            help='Default Null value is for Surfer grid. dfv=1.70141e+0')
    parser.add_argument('--datahdrlines',type=int,default=0,help='data header lines to skip.default=0')
    parser.add_argument('--dataresampleby',default=1,type=int,help='for resampling data file.default=1')
    parser.add_argument('--startid',type=float,default=1000.0,help='start id.dfv=1000')
    parser.add_argument('--timeconvert',action='store_true',default=False,
            help='dfv=False, i.e. input is t2w in msec to depth convert')
    parser.add_argument('--multiplier',required=False,type=float,default=1.0,help='Depth or Time multiplier. dfv=1.0')
    parser.add_argument('--segyxhdr',type=int,default=73,help='xcoord header.default=73')
    parser.add_argument('--segyyhdr',type=int,default=77,help='ycoord header. default=77')
    parser.add_argument('--xyscalerhdr',type=int,default=71,help='hdr of xy scaler to divide by.default=71')
    parser.add_argument('--outrange',nargs = 2,type=float,default=[0,10000],help='start end values of computed file, default= 0 10000')
    parser.add_argument('--outdir',help='output directory for created segy,default= same dir as input')
    parser.add_argument('--qcoefs',nargs=3,type=float,help='a0 a1 a2')
    parser.add_argument('--qcfilename',help=' One Quadratic coef flat file for all 3 coefs')
    parser.add_argument('--qccols',type=int, nargs=5,default=(1,2,6,7,8),
      help='5 values expected: columns for x y a0 a1 a2 dfv= 1 2 6 7 8')
    parser.add_argument('--qchdrlines',type=int,default=1,help='qcfile header lines. default=1')
    parser.add_argument('--qcresampleby',type=int,default=1,help='q coef lines to skip. default=1')
    parser.add_argument('--afiles',nargs=3,help='a0  a1 and a2 coef in 3 seperate file names.')
    parser.add_argument('--coeffiletype',choices=['xyz','zmap','surfer'],default='xyz',
            help='Input file type: flat ASCII, zmap grid, surfer grid. dfv= xyz ')
    parser.add_argument('--fqcols',nargs=9 ,type=int, default=(0,1,2,0,1,2,0,1,2),
            help='x y a0 x y a1 x y a2 in seperate ASCII flat files. dfv = 0 1 2 0 1 2 0 1 2')
    parser.add_argument('--ahdrlines',type=int,default=0,help='afiles header lines. default=0')
    parser.add_argument('--alinestoskip',default=1,type=int,help='for resampling a coefficients grid files only.default= 1')
    parser.add_argument('--agridmultipliers',nargs=3,type=float,default=(1.0,1.0,1.0),
            help='3 multipliers to a coefficients grid files. dfv= 1.0 1.0 1.0')


    result=parser.parse_args()
    if not result.datafilename:
        parser.print_help()
        exit()
    else:
        return result



def main():
    cmdl=getcommandline()
    #surfer grid in can have surfer grid out or zmap grid out


    if cmdl.datafiletype =='segy':
        segyfname = cmdl.datafilename
        dirsplit,fextsplit= os.path.split(cmdl.datafilename)
        fname,fextn= os.path.splitext(fextsplit)
        if cmdl.qcfilename:
            xyqc,qcoef=qcoefin(cmdl.qcfilename,cmdl.qccols,cmdl.qchdrlines,cmdl.qcresampleby)
            if cmdl.outdir:
                outfnamec = os.path.join(cmdl.outdir,fname) +"_ztq1f.sgy" # spatially varying q single file
                outvfnamec = os.path.join(cmdl.outdir,fname) +"_vavq1f.sgy" # spatially varying q single file
            else:
                outfnamec = os.path.join(dirsplit,fname) +"_ztq1f.sgy"
                outvfnamec = os.path.join(dirsplit,fname) +"_vavq1f.sgy" # spatially varying q single file
            start_copy = datetime.now()
            copyfile(segyfname, outfnamec)
            copyfile(segyfname, outvfnamec)
            end_copy = datetime.now()
            print('Duration of copying: {}'.format(end_copy - start_copy))
        elif cmdl.afiles: #3 coef files
            if cmdl.coeffiletype == 'surfer':  # 3 surfer grid coef files
                xy0,a0= surfergrid2xyz(cmdl.afiles[0],cmdl.nullvalue,cmdl.agridmultipliers[0],cmdl.alinestoskip)
                xy1,a1= surfergrid2xyz(cmdl.afiles[1],cmdl.nullvalue,cmdl.agridmultipliers[1],cmdl.alinestoskip)
                xy2,a2= surfergrid2xyz(cmdl.afiles[2],cmdl.nullvalue,cmdl.agridmultipliers[2],cmdl.alinestoskip)
            elif cmdl.coeffiletype == 'zmap':
                xy0,a0= zmapgridin(cmdl.afiles[0],cmdl.agridmultipliers[0],cmdl.alinestoskip)
                xy1,a1= zmapgridin(cmdl.afiles[1],cmdl.agridmultipliers[1],cmdl.alinestoskip)
                xy2,a2= zmapgridin(cmdl.afiles[2],cmdl.agridmultipliers[2],cmdl.alinestoskip)

            else:   #3 coef ASCII flat file
                xy0,a0=acoefin(cmdl.afiles[0],cmdl.fqcols[0],cmdl.fqcols[1],cmdl.fqcols[2],cmdl.ahdrlines,cmdl.alinestoskip)
                xy1,a1=acoefin(cmdl.afiles[1],cmdl.fqcols[3],cmdl.fqcols[4],cmdl.fqcols[5],cmdl.ahdrlines,cmdl.alinestoskip)
                xy2,a2=acoefin(cmdl.afiles[2],cmdl.fqcols[6],cmdl.fqcols[7],cmdl.fqcols[8],cmdl.ahdrlines,cmdl.alinestoskip)
            if cmdl.outdir:
                outfnamec = os.path.join(cmdl.outdir,fname) +"_ztq3f.sgy" #spatially varying q from 3 files
                outvfnamec = os.path.join(cmdl.dirsplit,fname) +"_vavq3f.sgy" #spatially varying q from 3 files
            else:
                outfnamec = os.path.join(dirsplit,fname) +"_ztq3f.sgy"
                outvfnamec = os.path.join(cmdl.dirsplit,fname) +"_vavq3f.sgy" #spatially varying q from 3 files
            print('Copying file, please wait ........')
            start_copy = datetime.now()
            copyfile(segyfname, outfnamec)
            copyfile(segyfname, outvfnamec)
            end_copy = datetime.now()
            print('Duration of copying: {}'.format(end_copy - start_copy))
        else: # cmdl.qcoef only 1 set of a0 a1 a2
            if cmdl.outdir:
                outfnamec = os.path.join(cmdl.outdir,fname) +"_zt1q.sgy"
                outvfnamec = os.path.join(cmdl.outdir,fname) +"_vav1q.sgy"
            else:
                outfnamec = os.path.join(dirsplit,fname) +"_zt1q.sgy"
                outvfnamec = os.path.join(dirsplit,fname) +"_vav1q.sgy"
            print('Copying file, please wait ........')
            start_copy = datetime.now()
            copyfile(segyfname, outfnamec)
            copyfile(segyfname, outvfnamec)
            end_copy = datetime.now()
            print('Duration of copying: {}'.format(end_copy - start_copy))

        with segyio.open(outfnamec, "r+" ,strict=False) as srcp, segyio.open(outvfnamec, "r+" ,strict=False) as srcv:
            mappedp = srcp.mmap()
            mappedv = srcv.mmap()
            if mappedp:
                print('Sucessful mapping to memory of {}'.format(outfnamec))
            else:
                print('Unable to map {} to memory'.format(outfnamec))
            if mappedv:
                print('Sucessful mapping to memory of {}'.format(outvfnamec))
            else:
                print('Unable to map {} to memory'.format(outvfnamec))
            nt = segyio.tools.sample_indexes(srcp)
            nta = np.array(nt)/1000
            # print(nta.min(),nta.max())
            ns = len(srcp.trace[1])
            # print(ns)
            zci,isample = np.linspace(cmdl.outrange[0],cmdl.outrange[1],num= ns,retstep =True)
            print('New sample interval: {}'.format(isample) )
            # print('len of zci',zci.size)
            print('min zci',zci.min(),'max zci ',zci.max())
            for trnum,tr in enumerate(srcp.trace):
                # print('Trace #: {}'.format(trnum))
                # print('Trace #: {} {}'.format(trnum,fb[trnum]))
                xysc = np.fabs(srcp.header[trnum][cmdl.xyscalerhdr])
                xc = srcp.header[trnum][cmdl.segyxhdr]/ xysc
                yc = srcp.header[trnum][cmdl.segyyhdr]/ xysc
                xy = np.array([xc,yc])
                if cmdl.qcoefs:
                    if cmdl.timeconvert:
                        tc,vav = tconv1xyz(xy,nta,cmdl.qcoefs)
                        ti1d = interpolate.interp1d(tc,tr,kind='cubic',bounds_error= False, fill_value =tr[0])
                        srcp.trace[trnum] = ti1d(zci).astype('float32')
                        ti1d = interpolate.interp1d(tc,vav,kind='cubic',bounds_error= False, fill_value =vav[-1])
                        srcv.trace[trnum] = ti1d(zci).astype('float32')

                    else:
                        zc,vav = zconv1xyz(xy,nta,cmdl.qcoefs)
                        zi1d = interpolate.interp1d(zc,tr,kind='cubic',bounds_error= False, fill_value =tr[0])
                        srcp.trace[trnum] = zi1d(zci).astype('float32')
                        zi1d = interpolate.interp1d(zc,vav,kind='cubic',bounds_error= False, fill_value =vav[-1])
                        srcv.trace[trnum] = zi1d(zci).astype('float32')
                elif cmdl.qcfilename:
                    if cmdl.timeconvert:
                        tc,vav,q0,q1,q2 = segytconvfile(xy,nta,xyqc,qcoef)
                        ti1d = interpolate.interp1d(tc,tr,kind='cubic',bounds_error= False, fill_value =tr[0])
                        srcp.trace[trnum] = ti1d(zci).astype('float32')
                        ti1d = interpolate.interp1d(tc,vav,kind='cubic',bounds_error= False, fill_value =vav[-1])
                        srcv.trace[trnum] = ti1d(zci).astype('float32')

                    else:
                        zc,vav,q0,q1,q2 = segyzconvfile(xy,nta,xyqc,qcoef)
                        zi1d = interpolate.interp1d(zc,tr,kind='cubic',bounds_error= False, fill_value =tr[0])
                        srcp.trace[trnum] = zi1d(zci).astype('float32')
                        zi1d = interpolate.interp1d(zc,vav,kind='cubic',bounds_error= False, fill_value =vav[-1])
                        srcv.trace[trnum] = zi1d(zci).astype('float32')
                elif cmdl.afiles: #3 coef files
                    if cmdl.timeconvert:
                        tc,vav,q0,q1,q2 = segytconvqfile(xy,nta,xy0,a0,xy1,a1,xy2,a2)
                        ti1d = interpolate.interp1d(tc,tr,kind='cubic',bounds_error= False, fill_value =tr[0])
                        srcp.trace[trnum] = ti1d(zci).astype('float32')
                        ti1d = interpolate.interp1d(tc,vav,kind='cubic',bounds_error= False, fill_value =vav[-1])
                        srcv.trace[trnum] = ti1d(zci).astype('float32')
                    else:
                        zc,vav,q0,q1,q2 = segyzconvqfile(xy,nta,xy0,a0,xy1,a1,xy2,a2)
                        zi1d = interpolate.interp1d(zc,tr,kind='cubic',bounds_error= False, fill_value =tr[0])
                        srcp.trace[trnum] = zi1d(zci).astype('float32')
                        zi1d = interpolate.interp1d(zc,vav,kind='cubic',bounds_error= False, fill_value =vav[-1])
                        srcv.trace[trnum] = zi1d(zci).astype('float32')

            isample *= 1000
            segyio.tools.resample(srcp,rate=int(isample),micro=True)
            segyio.tools.resample(srcv,rate=int(isample),micro=True)
            print('New sample interval: {},from converted file {}'.format(isample,segyio.tools.dt(srcp)) )
            print('New sample interval: {},from vav file {}'.format(isample,segyio.tools.dt(srcv)) )

    else:
        dirsplit,fextsplit= os.path.split(cmdl.datafilename)
        fname,fextn= os.path.splitext(fextsplit)
        if cmdl.datafiletype == 'surfer' :
            xy,t,ncols,nrows,minx,maxx,miny,maxy= surfergrid2xyz(cmdl.datafilename,cmdl.nullvalue,cmdl.multiplier,cmdl.dataresampleby)
            print("Number of non Null values %10d  " %(t.shape), file=sys.stderr)
            #print >> sys.stderr, xy.shape
        #zmap grid in can have zmap grid out or surfer grid out
        elif cmdl.datafiletype == 'zmap' :
            xy,t,ncols,nrows,minx,maxx,miny,maxy= zmapgridin(cmdl.datafilename,cmdl.multiplier,cmdl.dataresampleby)
            print("Number of non Null values %10d " %(t.shape), file=sys.stderr)
            #print >> sys.stderr, xy.shape

        elif cmdl.datafiletype =='xyz':  #xyz file in has to xyz file out
            xy,t=datain(cmdl.datafilename,cmdl.datacols,cmdl.multiplier,cmdl.datahdrlines,cmdl.dataresampleby)
            print("Number of non Null values %10d " %(t.shape), file=sys.stderr)

            if cmdl.timeconvert:
                horcols = ['X','Y','Z']
                hordf = pd.DataFrame({'X':xy[:,0],'Y':xy[:,1],'Z':t})
            else:
                horcols = ['X','Y','T1W','T2W']
                hordf = pd.DataFrame({'X':xy[:,0],'Y':xy[:,1],'T1W':t/ 2000.0,'T2W':t})
            hordf = hordf[horcols].copy()
        elif cmdl.datafiletype == 'kngdmflt' :
            xy,t2w,segid,fltname = datain_kngdmflt(cmdl.datafilename,cmdl.multiplier)

        if cmdl.qcoefs:
            if cmdl.datafiletype =='kngdmflt':
                if cmdl.timeconvert:
                    tc,vav = tconv1xyz(xy,t2w,cmdl.qcoefs,kflt=True)
                    fltfilename = fname +'_tconv.txt'
                    list_kngdmflt(fltfilename,xy,tc,segid,fltname)
                    print('Successfully saved {}'.format(fltfilename))
                else:
                    zc,vav = zconv1xyz(xy,t2w,cmdl.qcoefs,kflt=True)
                    fltfilename = fname +'_zconv.txt'
                    list_kngdmflt(fltfilename,xy,zc,segid,fltname)
                    print('Successfully saved {}'.format(fltfilename))
            else:
                if cmdl.timeconvert:
                    tc,vav = tconv1xyz(xy,t,cmdl.qcoefs)
                    hordf['T1W'] = tc
                    hordf['T2w'] = tc *2000.0
                    hordf['VAV'] = vav


                    hordf['T2W'] = hordf['T2W'].round(0)
                    hordf['T1W'] = hordf['T1W'].round(3)
                    hordf['VAV'] = hordf['VAV'].round(0)

                    tcfname = fname + '_tc.txt'
                    hordf.to_csv(tcfname,index=False,sep=' ')
                    print('Successfully saved {}'.format(tcfname))
                else:
                    zc,vav = zconv1xyz(xy,t,cmdl.qcoefs)
                    hordf['Z'] = zc
                    hordf['VAV'] = vav

                    hordf['Z'] = hordf['Z'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)

                    zcfname = fname + '_zc.txt'
                    hordf.to_csv(zcfname,index=False,sep=' ')
                    print('Successfully saved {}'.format(zcfname))

        # else:
           # listdatain(xy,t)
        if cmdl.qcfilename:
            xyqc,qcoef=qcoefin(cmdl.qcfilename,cmdl.qccols,cmdl.qchdrlines,cmdl.qcresampleby)

            if cmdl.datafiletype =='kngdmflt':
                if cmdl.timeconvert:
                    tc,vav = tconvfile(xy,t2w,xyqc,qcoef)
                    fltfilename = fname +'_tconv.txt'
                    list_kngdmflt(fltfilename,xy,tc,segid,fltname)
                    print('Successfully saved {}'.format(fltfilename))
                else:
                    zc,vav = zconvfile(xy,t2w,xyqc,qcoef)
                    fltfilename = fname +'_zconv.txt'
                    list_kngdmflt(fltfilename,xy,zc,segid,fltname)
                    print('Successfully saved {}'.format(fltfilename))
            else:


                if cmdl.timeconvert:
                    tc,vav,q0,q1,q2 = tconvfile(xy,t,xyqc,qcoef)
                    hordf['T1W'] = tc
                    hordf['T2w'] = tc * 2000.00
                    hordf['VAV'] = vav
                    hordf['QA0'] = q0
                    hordf['QA1'] = q1
                    hordf['QA2'] = q2

                    hordf['T1W'] = hordf['T1W'].round(3)
                    hordf['T2w'] = hordf['T2w'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)
                    hordf['T1W'] = hordf['T1W'].round(3)
                    hordf['T2w'] = hordf['T2w'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)
                    hordf['QA0'] = hordf['QA0'].round(0)
                    hordf['QA1'] = hordf['QA1'].round(0)
                    hordf['QA2'] = hordf['QA2'].round(0)


                    tcfname = fname + '_tcqf.txt'
                    hordf.to_csv(tcfname,index=False,sep=' ')
                    print('Successfully saved {}'.format(tcfname))

                else:
                    zc,vav,q0,q1,q2 = zconvfile(xy,t,xyqc,qcoef)
                    hordf['Z'] = zc
                    hordf['VAV'] = vav
                    hordf['QA0'] = q0
                    hordf['QA1'] = q1
                    hordf['QA2'] = q2



                    hordf['Z'] = hordf['Z'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)
                    hordf['QA0'] = hordf['QA0'].round(0)
                    hordf['QA1'] = hordf['QA1'].round(0)
                    hordf['QA2'] = hordf['QA2'].round(0)


                    zcfname = fname +'_zcqf.txt'
                    hordf.to_csv(zcfname,index=False,sep=' ')
                    print('Successfully saved {}'.format(zcfname))

        elif cmdl.afiles: #3 coef files

            if cmdl.coeffiletype == 'surfer':  # 3 surfer grid coef files
                xy0,a0= surfergrid2xyz(cmdl.afiles[0],cmdl.nullvalue,cmdl.agridmultipliers[0],cmdl.alinestoskip)
                xy1,a1= surfergrid2xyz(cmdl.afiles[1],cmdl.nullvalue,cmdl.agridmultipliers[1],cmdl.alinestoskip)
                xy2,a2= surfergrid2xyz(cmdl.afiles[2],cmdl.nullvalue,cmdl.agridmultipliers[2],cmdl.alinestoskip)
            elif cmdl.coeffiletype == 'zmap':
                xy0,a0= zmapgridin(cmdl.afiles[0],cmdl.agridmultipliers[0],cmdl.alinestoskip)
                xy1,a1= zmapgridin(cmdl.afiles[1],cmdl.agridmultipliers[1],cmdl.alinestoskip)
                xy2,a2= zmapgridin(cmdl.afiles[2],cmdl.agridmultipliers[2],cmdl.alinestoskip)

            else:   #3 coef ASCII flat file
                xy0,a0=acoefin(cmdl.afiles[0],cmdl.fqcols[0],cmdl.fqcols[1],cmdl.fqcols[2],cmdl.ahdrlines,cmdl.alinestoskip)
                xy1,a1=acoefin(cmdl.afiles[1],cmdl.fqcols[3],cmdl.fqcols[4],cmdl.fqcols[5],cmdl.ahdrlines,cmdl.alinestoskip)
                xy2,a2=acoefin(cmdl.afiles[2],cmdl.fqcols[6],cmdl.fqcols[7],cmdl.fqcols[8],cmdl.ahdrlines,cmdl.alinestoskip)

            if cmdl.datafiletype =='kngdmflt':
                if cmdl.timeconvert:
                    tc,vav,q0,q1,q2 = tconvqfile(xy,t2w,xy0,a0,xy1,a1,xy2,a2)
                    fltfilename = fname +'_tconv.txt'
                    list_kngdmflt(fltfilename,xy,tc,segid,fltname)
                    print('Successfully saved {}'.format(fltfilename))
                else:
                    zc,vav,q0,q1,q2 = zconvqfile(xy,t2w,xy0,a0,xy1,a1,xy2,a2)
                    fltfilename = fname +'_zconv.txt'
                    list_kngdmflt(fltfilename,xy,zc,segid,fltname)
                    print('Successfully saved {}'.format(fltfilename))
            else:


                if cmdl.timeconvert:
                    tc,vav,q0,q1,q2 = tconvqfile(xy,t,xy0,a0,xy1,a1,xy2,a2)
                    hordf['T1W'] = tc
                    hordf['T2w'] = tc * 2000.00
                    hordf['VAV'] = vav
                    hordf['QA0'] = q0
                    hordf['QA1'] = q1
                    hordf['QA2'] = q2

                    hordf['T1W'] = hordf['T1W'].round(3)
                    hordf['T2w'] = hordf['T2w'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)
                    hordf['T1W'] = hordf['T1W'].round(3)
                    hordf['T2w'] = hordf['T2w'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)
                    hordf['QA0'] = hordf['QA0'].round(0)
                    hordf['QA1'] = hordf['QA1'].round(0)
                    hordf['QA2'] = hordf['QA2'].round(0)



                    tcfname = fname + '_tcqf.txt'
                    hordf.to_csv(tcfname,index=False,sep=' ')
                    print('Successfully saved {}'.format(tcfname))
                else:
                    zc,vav,q0,q1,q2 = zconvqfile(xy,t,xy0,a0,xy1,a1,xy2,a2)
                    hordf['Z'] = zc
                    hordf['VAV'] = vav
                    hordf['QA0'] = q0
                    hordf['QA1'] = q1
                    hordf['QA2'] = q2

                    hordf['Z'] = hordf['Z'].round(0)
                    hordf['VAV'] = hordf['VAV'].round(0)
                    hordf['QA0'] = hordf['QA0'].round(0)
                    hordf['QA1'] = hordf['QA1'].round(0)
                    hordf['QA2'] = hordf['QA2'].round(0)



                    zcfname = fname +'_zcqf.txt'
                    hordf.to_csv(zcfname,index=False,sep=' ')
                    print('Successfully saved {}'.format(zcfname))







if __name__ == '__main__' :
    main()


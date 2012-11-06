from spatial import variogram as vm
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats

outfile = open("./ArizonaSim.csv", "w")

outfile.write("NH4N,X_COORD,Y_COORD,VegCover,percentclay\n")
n = 300
TrueBetaIntercept = 50
TrueBetaVeg = 2
TrueBetaClay = -0.5
TruePhi = 0.1
TrueTauSq = 50
TrueSigmaSq = 1
TrueErrorSquared = 10

vgm = vm.GaussianVariogram(TrueTauSq, TrueSigmaSq, TruePhi)

Beta = np.array([TrueBetaIntercept, TrueBetaVeg, TrueBetaClay])
Beta.shape = (3,1)

Data = np.zeros(n*(len(Beta) + 3))
Data.shape = (n, len(Beta) + 3)
veg = stats.gamma.rvs(10, size = n)
clay = stats.gamma.rvs(10, size = n)
intercept = np.array([1]*n)
eastings = stats.norm.rvs(loc = 0, scale = 10000, size = n)
northings = stats.norm.rvs(loc = 0, scale = 10000, size = n)
dists = squareform(pdist(np.vstack([eastings, northings]).transpose()))
X = np.vstack([intercept, veg, clay]).transpose()

Mu = X.dot(Beta) 
Mu.shape = (n,)


def f(tau, sigma, phi):
    vgm.tau_squared = tau
    vgm.sigma_squared = sigma
    vgm.phi = phi
    return(np.linalg.det(vgm.C(dists) + np.diag([tau]*n)))
import pdb; pdb.set_trace()
V = vgm.C(dists) + np.diag([TrueErrorSquared]*n)
Y = np.random.multivariate_normal(Mu.transpose(), V)
Y += stats.norm.rvs(loc=0, scale = 10**0.5, size = n)
out = np.vstack([Y, eastings, northings, veg, clay]).transpose()
for rw in xrange(out.shape[0]):
    outfile.write(",".join([str(x) for x in out[rw,:]]) + "\n")

outfile.close()

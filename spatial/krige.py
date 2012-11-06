import variogram as vm
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
import matplotlib.pyplot as plt

def metropolis_draw(variable, f, g, collapse = True):
    itrs = 0
    if not "__len__" in dir(variable):
        varlen = 1
    else:
        varlen = len(variable)

    while (itrs < 50):
        newvar = (g.rvs(varlen))
        ratio = np.sum(f(newvar)) - np.sum(f(variable))
        if (np.log(stats.uniform.rvs()) < ratio):
            return(newvar)
        itrs += 1
    return(variable)

def mvtnorm(Y, Mu, V):
    try:
        L = np.linalg.cholesky(V)
        logDet = np.sum(np.log(np.diag(L)))
    except:
        return(-np.infty)
    return((-len(Y)/2.0)*np.log(2*np.pi) - 0.5 * logDet + ((-0.5)*(Y-Mu).transpose().dot(np.linalg.inv(V).dot(Y-Mu))))

def mvtnorm_V(Y, Mu, logDetV, invV):
    return((-len(Y)/2.0)*np.log(2*np.pi)  - 0.5 * logDetV + ((-0.5)*(Y-Mu).transpose().dot(invV.dot((Y-Mu)))))

def mvtnorm_2(Y, Mu, V):
    try:
        L = np.linalg.cholesky(V)
        logDet = np.sum(np.log(np.diag(L)))
    except:
        return(-np.infty)
    return(0.5 * logDet + ((-0.5)*(Y-Mu).transpose().dot(np.linalg.inv(V).dot((Y-Mu)))))

def mvtnorm_3(Y, Mu, V):
    return((-0.5)*(Y-Mu).transpose().dot(np.linalg.inv(V).dot((Y-Mu))))

class bayes():
    def __init__(self, y, eastings, northings, X, vgm = "gaussian"):
        self.y = y
        self.loc = np.vstack([eastings, northings]).transpose()
        self.X = X        
        self.dists = squareform(pdist(self.loc))

        self.Beta = np.array([0]*X.shape[1])      
        self.Beta += np.random.normal(loc = np.array([0.0]*len(self.Beta)), scale = 10)

        TauSq = np.sqrt(np.mean(self.dists))
        SSQ = np.var(self.y)
        #phi = 1/(np.mean(self.dists))
        #phi = 0.00 + np.random.normal(loc = 0.0, scale = 0.1)
        phi = -0.4

        self.error_squared = 10
        #self.vgm.tau_squared = 10
        #self.vgm.sigma_squared = 0.1
        #self.vgm.phi = 1
        self.vgm = {"gaussian":vm.GaussianVariogram(TauSq, SSQ, phi)}[vgm]
        self.chain = np.hstack([self.Beta, self.vgm.tau_squared, self.vgm.sigma_squared, self.vgm.phi, self.error_squared])
        self.chain.shape = (len(self.Beta) + 4,)
        self.gen_grid()
        #self.fulldists = squareform(pdist(np.hstack(self.loc, self.Y0.loc)))

    def gen_grid(self, k = 10.0, X0 = None):
        minE = np.min(self.loc[:,0])
        minN = np.min(self.loc[:,1])
        maxE = np.max(self.loc[:,0])
        maxN = np.max(self.loc[:,1])
        Nstep = (maxN - minN)/k
        Estep = (maxE - minE)/k
        self.Y0_loc = np.zeros(k*k*2)
        self.Y0_loc.shape = (k*k, 2)
        rw = 0
        for Eastings in np.arange(minE, maxE, Estep):
            for Northings in np.arange(minN, maxN, Nstep):
                self.Y0_loc[rw,0] = Eastings
                self.Y0_loc[rw,1] = Northings
        if X0 == None:
            X0 = np.zeros(self.Y0_loc.shape[0]*self.X.shape[1])
            X0.shape = (self.Y0_loc.shape[0], self.X.shape[1])
            X0[:,0] += 1
        self.X0 = X0

    def BetaFC(self, Beta):
        Mu = self.X.dot(Beta)
        #V = self.vgm.sigma_squared*(self.vgm.f(self.dists)) + np.diag([1]*len(self.Y))*self.vgm.tau_squared
        #vfunc = np.vectorize(self.vgm.f)
        V = (self.vgm.C(self.dists)) + np.diag([self.error_squared]*self.y.shape[0])
        return((mvtnorm(self.y, Mu, V)) + stats.norm.logpdf(Beta, [0]*len(Beta), [5]*len(Beta)))

    def Y0FC(self, Y0):
        Y_combined = np.vstack([self.y, Y0])
        V_combined = (self.vgm.C(self.fulldists)) + np.diag([self.error_squared]*self.y.shape[0])
        # Grab pieces of V_combined and Mu from X and X0 to make full conditional

    def VarianceComponentsFC(self, TauSq, SSQ, phi):
        if (TauSq <= 0 or SSQ <= 0):
            return(-np.infty)
        self.vgm.tau_squared = TauSq
        self.vgm.sigma_squared = SSQ
        self.vgm.phi = phi
        V = (self.vgm.C(self.dists)) + np.diag([self.error_squared]*self.y.shape[0])
        Mu = self.X.dot(self.Beta)
        return(mvtnorm(self.y, Mu, V) + stats.gamma.logpdf(TauSq, 5) + stats.gamma.logpdf(SSQ, 10) + stats.norm.logpdf(phi, 0,0.5))

    def TauSqFC(self, TauSq):
        if TauSq <= 0:
            return(-np.infty)
        self.vgm.tau_squared = TauSq
        #vfunc = np.vectorize(self.vgm.f)
        V = (self.vgm.C(self.dists)) + np.diag([self.error_squared]*self.y.shape[0])
        Mu = self.X.dot(self.Beta)
        return((mvtnorm(self.y, Mu, V)) + stats.gamma.logpdf(TauSq, 5))

    def SSQFC(self, SSQ):
        if SSQ <= 0:
            return(-np.infty)
        self.vgm.sigma_squared = SSQ
        #vfunc = np.vectorize(self.vgm.f)
        V = (self.vgm.C(self.dists))+ np.diag([self.error_squared]*self.y.shape[0])
        Mu = self.X.dot(self.Beta)
        return((mvtnorm(self.y, Mu, V)) + stats.gamma.logpdf(SSQ, 100))

    def PhiFC(self, phi):
        self.vgm.phi = phi
        #vfunc = np.vectorize(self.vgm.f)
        V = (self.vgm.C(self.dists)) + np.diag([self.error_squared]*self.y.shape[0])
        Mu = self.X.dot(self.Beta)
        return((mvtnorm(self.y, Mu, V)) + stats.norm.logpdf(phi, 0,0.5))

    def ErrorFC(self, Error):
        Mu = self.X.dot(self.Beta)
        V = (self.vgm.C(self.dists)) + np.diag([self.error_squared]*self.y.shape[0])
        return((mvtnorm(self.y, Mu, V)) + stats.gamma.logpdf(Error, 10)) 

    def drawError(self):
        candidate = stats.norm(loc = self.error_squared, scale = 0.1)
        out = (metropolis_draw(self.error_squared, lambda x: self.ErrorFC(x), candidate))
        self.error_squared = out[0]
        return(out)

    def drawVarianceComponents(self):
        loc = np.array([self.vgm.tau_squared, self.vgm.sigma_squared, self.vgm.phi])
        candidate = stats.norm(loc, scale = np.array([0.5, 1, 0.1]))
        out = metropolis_draw(loc, lambda x: self.VarianceComponentsFC(x[0], x[1], x[2]), candidate)
        self.vgm.tau_squared = out[0]
        self.vgm.sigma_squared = out[1]
        self.vgm.phi = out[2]
        return(out)

    def drawTauSq(self, scale = 0.5):
        candidate = stats.norm(loc = self.vgm.tau_squared, scale=scale)
        out = (metropolis_draw(self.vgm.tau_squared, lambda x: self.TauSqFC(x), candidate))
        self.vgm.tau_squared = out[0]
        return(out)

    def drawSSQ(self, scale = 1):
        candidate = stats.norm(loc = self.vgm.sigma_squared, scale=scale)
        out = (metropolis_draw(self.vgm.sigma_squared, lambda x: self.SSQFC(x), candidate))
        self.vgm.sigma_squared = out[0]
        return(out)
    
    def drawPhi(self, scale = 0.2):
        candidate = stats.norm(loc = self.vgm.phi, scale=scale)
        out = metropolis_draw(self.vgm.phi, lambda x: self.PhiFC(x), candidate)
        self.vgm.phi = out
        return(out)

    def drawBeta2(self):
        SigmaInv = np.linalg.inv((self.vgm.C(self.dists)) + np.diag([self.error_squared]*self.y.shape[0]))
        PriorSigmaInv = np.diag([1/5.0]*len(self.Beta))
        V = np.linalg.inv(PriorSigmaInv + self.X.transpose().dot(SigmaInv.dot(self.X)))
        Mu = V.dot((SigmaInv.dot(self.X)).transpose()).dot(self.y)

        out = np.random.multivariate_normal(Mu.transpose(), V)
        self.Beta = out
        return(out)

    def drawBeta(self):
        candidate = stats.norm(loc = self.Beta, scale = 0.1)
        out = (metropolis_draw(self.Beta, lambda x: self.BetaFC(x), candidate))
        self.Beta = out
        return(out)

    def Converge(self):
        self.drawTauSq(scale = 100)
        self.drawSSQ(scale = 100)
        #self.drawPhi(scale =10)
        for i in xrange(300000):
            if (i % 1000 == 0):
                print("%i iterations so far." % i)
            #print("Draw Beta")
            self.drawVarianceComponents()
            self.drawBeta2()
            #print("Draw Phi")
            #self.drawPhi()
            #print("Draw TauSq")
            #self.drawTauSq()
            #print("Draw SSQ")
            #self.drawSSQ()
            #self.drawVarianceComponents()
            self.drawError()
            self.chain = np.vstack([self.chain, np.hstack([self.Beta, self.vgm.tau_squared, self.vgm.sigma_squared, self.vgm.phi, self.error_squared])])
        np.savetxt("output3.csv", self.chain, delimiter = ",")

    def plotchain(self, i):
        plt.figure(1)
        plt.plot(np.arange(1,self.chain.shape[0] + 1, 1), self.chain[:,i], "bo")
        plt.show()


    def krige(self):
        pass

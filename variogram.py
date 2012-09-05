import numpy as np
from laspy.file import File
from laspy.header import Header

def norm(vec):
    return(sum([x**2 for x in vec])**0.5)

class Variogram():
    def __init__(self):
        pass

    def f(self, t):
        pass

    def get_3d_grid(self, lb, ub, granularity = 0.1):

        n = ((ub-lb)/granularity)
        grid = np.zeros((n**2)*3)
        grid.shape = (n**2,3)
        idx = 0
        x = lb
        y = lb
        for i in xrange(int(n/2)):
            for j in xrange(int(n/2)):
                val = self.f(norm([x,y]))
                grid[idx,:] = (x,y,val)
                idx += 1
                grid[idx,:] = (-x,y,val)
                idx += 1
                grid[idx,:] = (x,-y,val)
                idx += 1
                grid[idx,:] = (-x,-y,val)
                idx += 1
                x += granularity
                y += granularity
        return(grid)

    def write_las_grid(self,filename, lb = 100, ub = 100, granularity = 0.1):
        out_file = File(filename, mode = "w", header = Header())
        out_file.scale = (1.0,1.0,1.0)
        grid = self.get_3d_grid(lb, ub, granularity)
        out_file.X = grid[:,0]
        out_file.Y = grid[:,1]
        out_file.Z = grid[:,2]
        out_file.intensity = grid[:,2]
        return(out_file)



class LinearVariogram(Variogram):
    def __init__(self, tau_squared, sigma_squared):
        self.tau_squared = tau_squared
        self.sigma_squared = sigma_squared

    def f(self, t):
        if (t <= 0):
            return(0)
        return(self.tau_squared + self.sigma_squared*t)


class SphericalVariogram(Variogram):
    def __init__(self, tau_squared, sigma_squared, phi):
        self.tau_squared = tau_squared
        self.sigma_squared = sigma_squared
        self.phi = phi

    def f(self, t):
        if (t <= 0 or t > (1/self.phi)):
            return(0)
        return(self.tau_squared + self.simga_squared*((3/2.0)*self.phi*t - 0.5*(self.phi*t)**3))


class ExponentialVariogram(Variogram):
    def __init__(self, tau_squared, sigma_squared, phi):
        self.tau_squared = tau_squared
        self.sigma_squared = sigma_squared
        self.phi = phi
        
    def f(self, t):
        if (t <= 0):
            return(0)
        return(self.tau_squared + self.sigma_squared*(1-np.exp(-1*self.phi*t)))


class PoweredExponentialVariogram(Variogram):
    def __init__(self, tau_squared, sigma_squared, phi, p):
        self.tau_squared = tau_squared
        self.sigma_squared = sigma_squared
        self.phi = phi
        self.p = p
        
    def f(self, t):
        if (t <= 0):
            return(0)
        return(self.tau_squared + self.sigma_squared*(1-np.exp(-1*(np.abs(self.phi*t)**self.p))))


class GaussianVariogram(Variogram):
    def __init__(self, tau_squared, sigma_squared, phi):
        self.tau_squared = tau_squared
        self.sigma_squared = sigma_squared
        self.phi = phi

    def f(self, t):
        if (t <= 0):
            return(0)
        return(self.tau_squared + self.sigma_squared*(1-np.exp(-1*(self.phi**2)*(t**2))))





























































































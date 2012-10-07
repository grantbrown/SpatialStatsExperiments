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
        x = lb + granularity
        y = lb + granularity
        for i in xrange(int(n/2)):
            y = lb
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
                y += granularity
            x += granularity
        return(grid)

    def write_las_grid(self,filename, lb = -10, ub = 10, granularity = 0.1):
        out_file = File(filename, mode = "w", header = Header())
        out_file.header.scale = (0.001,0.001,0.001)
        grid = self.get_3d_grid(lb, ub, granularity)
        out_file.x = grid[:,0]* 300
        out_file.y = grid[:,1] * 300
        out_file.z = grid[:,2] * 300
        out_file.intensity = grid[:,2] * 300
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
        return(self.tau_squared + self.sigma_squared*((3/2.0)*self.phi*t - 0.5*(self.phi*t)**3))


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

if __name__ == "__main__":
    print("Writing Gaussian Variogram")
    gauss = GaussianVariogram(10,10,10)
    outfile = gauss.write_las_grid("gaussian.las", -1, 1, 0.01)
    outfile.close()
    print("Writing Powered Exponential Variogram")
    pexp = PoweredExponentialVariogram(10,10,10,1)
    outfile = pexp.write_las_grid("powered_exp.las", -1, 1, 0.01)
    outfile = outfile.close()
    print("Writing Exponential variogram")
    exp = ExponentialVariogram(10,10,10)
    outfile = exp.write_las_grid("exp.las", -1, 1, 0.01)
    outfile.close()
    print("Writing Spherical Variogram")
    spherical = SphericalVariogram(10,10,0.5)
    outfile = spherical.write_las_grid("sphere.las", -1,1,0.01)
    outfile.close()
    print("Writing Linear Variogram")
    linear = LinearVariogram(10,10)
    outfile = linear.write_las_grid("linear.las", -1,1,0.01)
    outfile.close()

























































































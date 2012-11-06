from spatial import krige as krige
import numpy as np

data = np.genfromtxt("./Arizona.csv", dtype = None, delimiter =",", names = True)


X = np.vstack([np.array([1]*data.shape[0]) ,data["VegCover"], data["percentclay"]]).transpose()

X.shape = (len(X), 3)
Y = data["NH4N"]
Eastings = data["X_COORD"]
Northings = data["Y_COORD"]

good = np.where(1 - ((np.isnan(X[:,0])) +
    (np.isnan(X[:,1])) +
    (np.isnan(X[:,2])) + 
    (np.isnan(Y)) +
    (np.isnan(Eastings)) +
    (np.isnan(Northings))))

Y = Y[good]
X = X[good]
Eastings = Eastings[good]
Northings = Northings[good]




k = krige.bayes(Y, 
                Eastings, 
                Northings, 
                X
                #np.vstack([data["Monitor_Classification"], data["Monitor_Radius_km"]]).transpose()
                )

#import cProfile
#cProfile.run("k.Converge()")
k.Converge()


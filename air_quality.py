from spatial import krige as krige
import numpy as np

data = np.genfromtxt("./chicagomondataR.csv", dtype = None, delimiter =",", names = True)

X = np.array([1]*data.shape[0]) 
X.shape = (len(X), 1)
k = krige.bayes(data["PM25_ugm3"], 
                data["EW_Coord_km"], 
                data["NS_Coord_km"], 
                X
                #np.vstack([data["Monitor_Classification"], data["Monitor_Radius_km"]]).transpose()
                )

k.Converge()

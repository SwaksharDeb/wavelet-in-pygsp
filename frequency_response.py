from pygsp import plotting
from pygsp import graphs, filters, plotting, utils
import numpy as np
from numpy.linalg import norm
from numpy import arccos, clip, dot

G = graphs.Logo()
mh = filters.MexicanHat(G, Nf=6)
plotting.plot_filter(mh)


G = mh.G
x = np.linspace(0, G.lmax)
y = mh.evaluate(x).T

# compute cosine similarity
A = y[:,0]
B = y[:,-1]
cosine = np.dot(A,B)/(norm(A)*norm(B))
angle = arccos(clip(cosine, -1, 1))
angle_degree = (angle*180)/3.1416

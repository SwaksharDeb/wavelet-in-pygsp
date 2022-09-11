from pygsp import plotting
from pygsp import graphs, filters, plotting, utils
import numpy as np

G = graphs.Logo()
mh = filters.MexicanHat(G, Nf=1)
plotting.plot_filter(mh)



G = mh.G
x = np.linspace(0, G.lmax, 1000)
y = mh.evaluate(x).T

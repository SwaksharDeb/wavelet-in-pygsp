import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils
import pygsp
import numpy as np
from pygsp import graphs, reduction

G = graphs.Sensor(512, distribute=True)
G.compute_fourier_basis()

levels = 5
Gs = reduction.graph_multiresolution(G, levels, sparsify=False)

data = Gs[-1].mr # to find which vertex has been kept in the last level and track the graph vertex along the level 

level_2_adj = Gs[1].A.todense()
adj = level_2_adj.astype(int)

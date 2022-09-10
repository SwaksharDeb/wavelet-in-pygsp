import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils
import pygsp
"""
G = graphs.Bunny()
tolerance = 0.00001

taus = [-1,-0.5,0.5,1,1.5,2]
g = filters.Heat(G, taus)

s = np.zeros(G.N)
DELTA = 20
s[DELTA] = 1

s_ = s

#s = g.filter(s, method='chebyshev')

chebyshev = pygsp.filters.approximations.compute_cheby_coeff(g, m=3)

wavelet_coefficients = pygsp.filters.approximations.cheby_op(G, chebyshev, s_)

wavelet_coefficients[wavelet_coefficients < tolerance] = 0

ind_1 = wavelet_coefficients.nonzero()

fig = plt.figure(figsize=(10, 3))
for i in range(g.Nf):
     ax = fig.add_subplot(1, g.Nf, i+1, projection='3d')
     G.plot_signal(s[:, i], colorbar=False, ax=ax)
     title = r'Heat diffusion, $\tau={}$'.format(taus[i])
     _ = ax.set_title(title)
     ax.set_axis_off()
fig.tight_layout()

g = filters.Heat(G, taus)  

fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of mexican hat wavelets')



import numpy as np 
import pygsp

W = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
G = pygsp.graphs.Graph(W)
heat_filter = pygsp.filters.Heat(G, tau=1)
chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=4)
impulse = np.eye(3, dtype=int)
wavelet_coefficients = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
wavelet_coefficients[wavelet_coefficients < 0.1] = 0
ind_1, ind_2 = wavelet_coefficients.nonzero()


import matplotlib.pyplot as plt
G = graphs.Ring(N=2)
G.estimate_lmax()
G.set_coordinates('line1D')
g = filters.MexicanHat(G)
s = g.localize(G.N // 2)
fig, axes = plt.subplots(1, 2)
g.plot(ax=axes[0])
G.plot_signal(s, ax=axes[1])


G = graphs.Sensor(30, seed=42)
G.compute_fourier_basis()  # Reproducible computation of lmax.
s1 = np.zeros(G.N)
s1[13] = 1
s1 = filters.Heat(G, 3).filter(s1)

g = filters.MexicanHat(G, Nf=4)
s2 = g.analyze(s1)
s2.shape
s2 = g.synthesize(s2)



import math
import numpy as np
G = graphs.Logo()
my_filter = filters.Filter(G, lambda x: x*np.exp(-x))
#my_filter = filters.Filter(G, lambda x: x / (1. + x))
#my_filter = filters.Filter(G, lambda x: 5./(5 + x))

# Signal: Kronecker delta.
signal = np.zeros(G.N)
signal[42] = 1
filtered_signal = my_filter.filter(signal)
filtered_signal[filtered_signal < 0.01] = 0

fig, ax = plt.subplots()
my_filter.plot(plot_eigenvalues=True, ax=ax)
_ = ax.set_title('Filter frequency response')


g = filters.MexicanHat(G, Nf=2)  # Nf = 6 filters in the filter bank.
fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of mexican hat wavelets')

"""
import numpy as np
from pygsp import graphs, reduction

G = graphs.Sensor(512, distribute=True)
G.compute_fourier_basis()

levels = 5
Gs = reduction.graph_multiresolution(G, levels, sparsify=False)

data = Gs[-1].mr # to find which vertex has been kept in the last level and track the graph vertex along the level 

level_2_adj = Gs[1].A.todense()
adj = level_2_adj.astype(int)

for gr in Gs:
    gr.compute_fourier_basis()

f = np.ones((G.N))
f[np.arange(G.N//2)] = -1
f = f + 10*Gs[0].U[:, 7]

f2 = np.ones((G.N, 2))
f2[np.arange(G.N//2)] = -1

g = [lambda x: x*np.exp(-x)]

ca, pe = reduction.pyramid_analysis(Gs, f, h_filters=g, method='exact')
ca2, pe2 = reduction.pyramid_analysis(Gs, f2, h_filters=g, method='exact')

#f_interpolated = reduction.interpolate(G, Gs[1], keep_inds=ca[1])
f_interpolated = reduction.interpolate(Gs[4], ca[5], Gs[5].mr['idx'])
f_interpolated_final = f_interpolated + pe[4]
#f_pred, _ = reduction._pyramid_single_interpolation(Gs[0], ca[1], pe[1], keep_inds=pe[0], h_filter=g, method='exact')

f_pred, _ = reduction.pyramid_synthesis(Gs, ca[levels], pe, method='exact')
f_pred2, _ = reduction.pyramid_synthesis(Gs, ca2[levels], pe2, method='exact')


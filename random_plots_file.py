import numpy as np
import h5py
import matplotlib.pyplot as plt

file = h5py.File('data/dgraph0.2/Mgraph_0098.hdf5','r')

nprogs = file['nProgs']
bins = int(np.sqrt(nprogs.size))
hist, bin_edges = np.histogram(file['nProgs'], bins = bins)
bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
plt.bar(bin_cents, hist)
plt.xlabel('bins')
plt.ylabel('number')
plt.show()
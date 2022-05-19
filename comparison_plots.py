import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import log10, floor

ll = [0.16, 0.25]
markers = ['.', 'v']
f0 = h5py.File('data/halos-0.16/halos_0050.hdf5','r')
f1 = h5py.File('data/halos-0.2/halos_0050.hdf5','r')
f2 = h5py.File('data/halos-0.25/halos_0050.hdf5','r')


files = [f0, f2]
line_colors = ["black","red", "blue"]
plt.figure(0)

for i in files: 
    
    n_parts = i['nparts']
    

    pmass = i.attrs['part_mass']

    mass = n_parts * pmass

    bins = np.logspace(10, 14, 50)

    H, bin_edges = np.histogram(mass, bins=bins)

    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(bin_cents, H, label = str(ll[files.index(i)]), color = line_colors[files.index(i)])
    

plt.title('$Mass Function$')
plt.xlabel('$Mass$')
plt.ylabel('$Number$')
plt.legend(title = 'Linking Length')
plt.title('$Mass\;\; Function\;\; -\;\; Red\;\; shift\;\; ~\;\; 1$')
#plt.savefig('report_plots/comparison_plots/mass_func_comp.png')
plt.show()


plt.figure(1)
for i in files:
    
    threeD_vel = i['3D_velocity_dispersion']
    v_max = i['v_max']
    

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$v_{max}$')
    plt.xlabel("$\sigma_{3D}$")
    plt.scatter(threeD_vel, v_max, s = 2, label = str(ll[files.index(i)]), color = line_colors[files.index(i)], marker = markers[files.index(i)])
    m, b = np.polyfit(np.log10(threeD_vel), np.log10(v_max), 1)
    x = np.logspace(1.5,2.7)
    plt.plot(x, 10**(m*np.log10(x) + b))

plt.title('$V_{max}\;\; Vs\;\; \sigma_{3D}\;\; - \;\;Comparison$')
plt.legend(title = 'Linking Length')
plt.show()
#plt.savefig('report_plots/comparison_plots/3Dvel_vs_vmax.png')

plt.figure(2)
for i in files:
    n_parts1 = i['nparts']

    pmass1 = i.attrs['part_mass']

    mass1 = pmass1 * n_parts1
    
    half_mass_radius = i['half_mass_radius']
    
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('$Mass$')
    plt.ylabel('$Half\;Mass\;Radius$')
    plt.scatter(mass1, half_mass_radius, s = 2, label = str(ll[files.index(i)]), color = line_colors[files.index(i)], marker = markers[files.index(i)])

plt.legend(title = 'Linking Length')
plt.title('$Half\;\; Mass\;\; Radius\;\; Vs\;\; Mass\;\; - \;\;Comparison$')
plt.show()
#plt.savefig('report_plots/comparison_plots/hmr_vs_mass.png')
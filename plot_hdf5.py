import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#want 10 50 70 - plot all mass functions, (redshift as legend)
link_lengths = ['0.16','0.2','0.25']
plt.figure(0)
for ll in link_lengths:
    f = h5py.File('data/halos-'+ll+'/halos_0098.hdf5','r')
    file_keys = list(f.keys())
    #print(file_keys)
    file_attriubutes = f.attrs.keys()
    #print('\n\n',file_attriubutes)


    n_parts = f['nparts']

    pmass = f.attrs['part_mass']

    mass = pmass * n_parts

    red_shift = f.attrs['redshift']

    bins = np.logspace(10, 14, 30)

    H, bin_edges = np.histogram(mass, bins=bins)


    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mass')
    plt.ylabel(r'$N$')
    plt.plot(bin_cents, H, label = ll)


plt.legend(title='Linking length')

plt.title('Mass Function Comparison')
plt.savefig('report_plots/comparison_plots/mass_func.png')



# threeD_vel = f['3D_velocity_dispersion']
# v_max = f['v_max']
# half_mass_radius = f['half_mass_radius']
# plt.figure(2)
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('$v_{max}$')
# plt.xlabel("$\sigma_{3D}$")
# plt.scatter(threeD_vel, v_max, s = 3)
# plt.title('Linking Length (0.2)')

# plt.figure(3)
# plt.xscale('log')

# plt.xlabel('$Mass$')
# plt.ylabel('$Half\;Mass\;Radius$')
# plt.scatter(mass, half_mass_radius, s = 3)
# plt.title('Linking Length (0.2)')
# plt.show()

for n in range(98, -1, -1):
    print(n)

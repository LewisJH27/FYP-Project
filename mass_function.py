import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import log10, floor




def mass_func():
    colours = ['blue', 'orange', 'green']
    link_len = ['0.4', '0.2', '0.5']
    for i in range(len(link_len)):
        f0 = h5py.File('data/halos-' + link_len[i] + '/halos_0098.hdf5','r')
        plt.figure(0)
        n_parts = f0['nparts']
        pmass = f0.attrs['part_mass']
        mass = n_parts * pmass
        bins = np.logspace(10, 14, 50)
        H, bin_edges = np.histogram(mass, bins=bins)
        bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.xscale('log')
        plt.yscale('log')
        plt.plot(bin_cents, H, label = link_len[i], color=colours[i])

    plt.grid()
    plt.title('$Mass Function$')
    plt.xlabel('$Mass$')
    plt.ylabel('$Number$')
    plt.legend(title = 'Linking Length')
    plt.title(r'Mass Function - Large linking lengths')
    plt.savefig('report_plots/comparison_plots/mf_comp_large_lls.png')

mass_func()


def sub_halo_mf():
    sub_ll = ['0.08', '0.1', '0.126']
    for ll in sub_ll:
        f0 = h5py.File('data/halos_sub_' + ll + '/halos_0098.hdf5','r')
        
        plt.figure(0)
        n_parts = f0['Subhalos']['nparts']

        pmass = f0.attrs['part_mass']

        mass = n_parts * pmass

        bins = np.logspace(10, 14, 20)

        H, bin_edges = np.histogram(mass, bins=bins)

        bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.xscale('log')
        plt.yscale('log')
        plt.plot(bin_cents, H, label = ll)

    plt.title('$Mass Function$')
    plt.xlabel('$Mass$')
    plt.ylabel('$Number$')
    plt.legend(title = 'Sub Halo linking length')
    plt.title(r'Mass Function - Subhalos')
    plt.savefig('report_plots/sub_plots/mf_subhalo_comp_20bins.png')




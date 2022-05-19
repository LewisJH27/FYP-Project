import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from matplotlib.colors import LogNorm



import h5py
import seaborn as sns
import yaml
from core import utilities


ll = str(0.16)

sns.set_style("whitegrid")

def energy_plot():
    


    total_KE = []
    total_GE = []
    mass = []

    #Read param file
    paramfile = sys.argv[1]
    inputs, flags, params, cosmology, simulation = utilities.read_param(paramfile)

    #load snapshot list
    snaplist = list(np.loadtxt(inputs['snapList'], dtype = str))

    halo_sub = int(sys.argv[2])

    
    snap = '0098'
    pmass = 1

    # Loop through Merger Graph data assigning each value to the relevant list
    for snap in snaplist:

        

        #Create file to store results
        hdf = h5py.File(inputs['haloSavePath'] + 'halos_' + str(snap) + '.hdf5','r')

        pmass = hdf.attrs['part_mass']

        if halo_sub == 0:


            KE = hdf["halo_kinetic_energies"][...]
            GE = hdf["halo_gravitational_energies"][...]
            m = hdf['nparts'][...] * pmass
            reals = hdf['real_flag'][...]
            total_KE.extend(KE[reals])
            total_GE.extend(GE[reals])
            mass.extend(m[reals])


        else:

            # Get the number of progenitors and descendants
            KE = hdf["Subhalos"]["halo_kinetic_energies"][...]
            GE = hdf["Subhalos"]["halo_gravitational_energies"][...]
            m = hdf["Subhalos"]["nparts"][...] * pmass
            reals = hdf["Subhalos"]["real_flag"][...]
            total_KE.extend(KE[reals])
            total_GE.extend(GE[reals])
            mass.extend(m[reals])

        hdf.close()

        try:

            #set up plot
            fig = plt.figure(figsize=(8,6))
            axtwin = fig.add_subplot(111)
            ax = axtwin.twiny()

            #Plot data
            axtwin.scatter(m, KE / GE, facecolors="none", edgecolors="none")
            cbar = ax.hexbin(m / pmass, KE/GE, gridsize=50, xscale='log', yscale='log', mincnt=1,
                            norm=LogNorm(), linewidths=0.2)
            ax.axhline(1.0, linestyle='--', color='k', label='$E=0$')
            ax.axhline(0.5, linestyle='==', color='g', label='$2KE=|GE|$')

            axtwin.grid(False)
            axtwin.grid(True)

            #label axes
            ax.set_ylabel(r'$\mathrm{KE}/|\mathrm{GE}|$')
            axtwin.set_xlabel(r'$M_{h} / M_{\odot}$')
            ax.set_xlabel(r'$N_{p}$')

            #set scale
            ax.set_xscale("log")
            axtwin.set_xscale("log")
            ax.set_yscale("log")

            #set limits
            ax.set_ylim(10 ** -2, None)

            #get and draw legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            fig.colorbar(cbar, ax=ax)

            #save figure
            if halo_sub == 0:
                fig.savefig('analytics/energy_plots/halo_energies_' + snap + 'ke_test_new.png', bbox_inches='tight')
            else:
                fig.savefig('analytics/energy_plots/subhalo_energies_' + snap + 'ke_test_new.png', box_inches='tight')

            plt.close(fig)

        except ValueError:
            print('uh oh')

    total_KE = np.array(total_KE)
    total_GE = np.array(total_GE)
    mass = np.array(mass)

    ratio = total_KE / total_GE
    print("There are: ", ratio[ratio >1].size, " unbound halos")
    
    #set up plot
    fig = plt.figure(figsize=(8, 6))
    axtwin = fig.add_subplot(111)
    ax = axtwin.twiny()

    #plot data
    axtwin.scatter(mass, ratio, facecolors="none", edgecolors="none")
    cbar = ax.hexbin(mass / pmass, ratio, gridsize=50, xscale="log", yscale="log", mincnt=1,
                     norm=LogNorm(), linewidths=0.2)
    ax.axhline(1.0, linestyle="--", color="k", label="$E=0$")
    ax.axhline(0.5, linestyle="--", color="g", label="$2KE=|GE|$")

    axtwin.grid(False)
    axtwin.grid(True)

    #label axis
    axtwin.set_ylabel(r"$\mathrm{KE}/|\mathrm{GE}|$")
    axtwin.set_xlabel(r"$M_{h} / M_{\odot}$")
    ax.set_xlabel(r"$N_{p}$")

    # Set scale
    ax.set_xscale("log")
    axtwin.set_xscale("log")
    ax.set_yscale("log")

    # Set limits
    # ax.set_ylim(10 ** -2, None)

    # Get and draw legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.colorbar(cbar, ax=ax)
    fig.suptitle('Energy Plot - 0.5 Linking Length')

    # Save figure
    if halo_sub == 0:
        fig.savefig("analytics/energy_plots/halo_energies_0.5.png", bbox_inches="tight")
    else:
        fig.savefig("analytics/energy_plots/subhalo_energies_0.5.png", bbox_inches="tight")

    plt.close(fig)

    return






if __name__ == "__main__":

    energy_plot()
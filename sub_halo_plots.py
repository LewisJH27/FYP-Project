from re import sub
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from collections import defaultdict
import time

start_time = time.time()






def extract_subhalos():
    """
    Function to extract the subhalos and calculate the mass fraction for the
    central subhalo and satellite subhalos
    """


    all_subs = []
    cent_mass_frac = []
    sat_mass_frac = []


    for n in range(98, -1, -1):
        if n < 10:
            n = '0' + str(n)
        else:
            n = str(n)
        hdf = h5py.File('data/halos_sub_0.1/halos_00' + n + '.hdf5','r')
        pmass = hdf.attrs['part_mass']


        host_halos = hdf['Subhalos']['host_IDs'][...]
        for halo in host_halos:
            n_parts = hdf['nparts'][halo]
            host_mass = pmass * n_parts
            sub_halos = np.where(host_halos == halo)[0][...]
            # all_subs.extend(sub_halos)
            sub_radius = []
            for sub in sub_halos:
                sub_pos = hdf['Subhalos']['mean_positions'][sub]
                halo_pos = hdf['mean_positions'][halo]
                sub_radius.append(((halo_pos[0] - sub_pos[0])**2 + (halo_pos[1] - sub_pos[1])**2 + (halo_pos[2] - sub_pos[2])**2))
            min_radius = np.argmin(sub_radius)
            central_sub = sub_halos[min_radius]
            cent_mass = pmass * hdf['Subhalos']['nparts'][central_sub]
            # cent_subs.append(central_sub)
            sat_sub = np.delete(sub_halos, min_radius)
            if sat_sub.any() != 0:
                sat_mass = pmass * hdf['Subhalos']['nparts'][sat_sub]
                sat_mass_frac.extend(sat_mass/host_mass)
            # sat_subs.extend(sat_sub)
            cent_mass_frac.extend(cent_mass/host_mass)
        hdf.close()
    
    plt.figure()
    fracs = [cent_mass_frac, sat_mass_frac]
    legends = ['Central Sub Halos', 'Satelite Sub Halos' ]
    for i in range(len(fracs)):   
        H, bin_edges = np.histogram(fracs[i], bins=125, range=(0,1))
        bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.plot(bin_cents, H, label = legends[i])
    
    plt.yscale('log')
    plt.legend()
    plt.show()
# extract_subhalos()



         
    


def sub_halo_vsd():
    """
    Function to extract all halos from all snapshots and calculate the velocity-space density
    """

    relative_speed = []
    for i in range(98, -1, -1):
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)

        hdf = h5py.File('data/halos_sub_0.1/halos_00' + i + '.hdf5', 'r')
        host_halos = np.unique(hdf['Subhalos']['host_IDs'])
        # print(host_halos)
        # print(hdf['Subhalos'].keys())
        for halo in host_halos:
            vel_dis = hdf['rms_velocity_radius'][halo]
            sub_halos = np.where(host_halos == halo)[0][...]
            # all_subs.extend(sub_halos)
            sub_radius = []
            for sub in sub_halos:
                if hdf['Subhalos']['nparts'][sub] >= 10:
                    sub_pos = hdf['Subhalos']['mean_positions'][sub]
                    halo_pos = hdf['mean_positions'][halo]
                    sub_radius.append(((halo_pos[0] - sub_pos[0])**2 + (halo_pos[1] - sub_pos[1])**2 + (halo_pos[2] - sub_pos[2])**2))
            min_radius = np.argmin(sub_radius)
            central_sub = sub_halos[min_radius]
            cent_sub_speed = hdf['Subhalos']['mean_velocities'][central_sub]
            host_speed = hdf['mean_velocities'][halo]
            relative_speed.append((host_speed - cent_sub_speed)/vel_dis)
    plt.figure()
       
    H, bin_edges = np.histogram(relative_speed, bins=50)
    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    plt.plot(bin_cents, H, label = 'Central Sub Halo')
    plt.yscale('log')
    plt.legend()
    plt.show()       
    
    print('Execution time: ' + str(time.time() - start_time))

# sub_halo_vsd()

def sub_halo_sd():

    centrality = []
    for i in range(98, -1, -1):
        if i < 10:
            i = '0' + str(i)
        else:
            i = str(i)

        hdf = h5py.File('data/halos_sub_0.1/halos_00' + i + '.hdf5', 'r')
        host_halos = np.unique(hdf['Subhalos']['host_IDs'])
        
        for halo in host_halos:
            
            sub_halos = np.where(host_halos == halo)[0][...]

            sub_radius = []
            for sub in sub_halos:
                sub_pos = hdf['Subhalos']['mean_positions'][sub]
                halo_pos = hdf['mean_positions'][halo]
                sub_radius.append(((halo_pos[0] - sub_pos[0])**2 + (halo_pos[1] - sub_pos[1])**2 + (halo_pos[2] - sub_pos[2])**2))
            min_radius = np.argmin(sub_radius)
            central_sub = sub_halos[min_radius]
            centrality.append(min_radius/hdf['rms_spatial_radius'][halo])

    plt.figure()
       
    H, bin_edges = np.histogram(centrality, bins=50)
    bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    plt.plot(bin_cents, H, label = 'Central Sub Halo')
    plt.legend()
    plt.show()       
    
    print('Execution time: ' + str(time.time() - start_time))
sub_halo_sd()
   
"""
mean positions are centre of mass
rms spatial radius is what we need for sigma r


need mean velocities
rms velocity radius is what we need for sigma v

"""

# def vel_space_dens():















from enum import auto
import h5py
import statistics
import numpy as np
import astropy
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt
from matplotlib import rc




def get_times(ll):
    #Define an array to store redshifts
    redshift_array = []

    #loop through the 98 snapshots and extract all redshifts of each one to the array
    for i in range(98, -1, -1):
            
            snapshot_num = str(i)
            if len(snapshot_num) == 1:
                file_number = '0' + snapshot_num
                #print(file_number
            if len(snapshot_num) == 2:
                file_number = snapshot_num
                #print(file_number)
            
            f = h5py.File('data/halos-'+ ll + '/halos_00'+file_number+'.hdf5','r')

            red_shift = f.attrs['redshift']
            redshift_array.append(red_shift)

    #Define an array for the times
    cosmo_time = []

    #Use astropy to calculate the time corresponding to a given redshift     
    for n in redshift_array:

        cosmo_time.append((cosmo.age(n)).value[0])
    
    #Append the snapshot and its corresponding time to a dictionary to be used in the formula for logarithmic mass growth.
    dicts = {}
    snaplist = list(np.loadtxt('snaplist3.txt', dtype=str))

    for i in range(len(cosmo_time)):
        
        dicts[snaplist[i]] = cosmo_time[i]

    return dicts

def log_mass_growth():
    
    link_len = ['0.16','0.2', '0.25']
    plt.figure()
    for ll in link_len:
        lmg_array = []

        #Function to retrieve the time stamp for each snapshot given by the redshift.
        times = get_times(ll)
        
        for i in range(98, -1, -1):
            
            if i < 10:
                file = h5py.File("data/dgraph" + ll + "/Mgraph_00" + str(i) + "0" + ".hdf5", 'r')
            else:
                file = h5py.File("data/dgraph" + ll + "/Mgraph_00" + str(i) + ".hdf5", 'r')
            #Loop down the main branch and find the halo and progenitors mass and calculate the lmg using them.
            nparts = file['nparts']
            nprogs = file['nProgs']        
            prog_start_index = file['prog_start_index']
            progids = file['Prog_haloIDs']
            prog_nparts = file['Prog_nPart']
            
            
            for halo in file['halo_IDs'][:]:
                halo_nparts = file['nparts'][halo]
                if halo_nparts >= 1000:
                        
                        
                    nprog = nprogs[halo]
                    if nprog <= 0:
                        continue
                        
                    prog_masses = prog_nparts[prog_start_index[halo] : prog_start_index[halo] + nprogs[halo]]
                    
                    #open the halos folders for each snapshot and extract the halo and prog masses
                    halo_mass = file['nparts'][halo]

                    # Append the values of the logarithmic mass growth to an array to be plotted
                    for pm in prog_masses:
                                    
                        lmg = ((times[str(i)]+times[str(i-1)])*(halo_mass - pm))/((times[str(i)]-times[str(i-1)])*(halo_mass + pm))
                        lmg_array.append(lmg)
                else:
                    continue
            file.close()
                
        #Create an array of x values according to the formula for beta         
        x_values = []
        for l in lmg_array:
            x_values.append((np.arctan(l))/(np.pi/2))
        
        # Plot the results 
        x = sorted(x_values)
        
        H, bin_edges = np.histogram(x, bins=50)
        bin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.semilogy()
        plt.plot(bin_cents, H, label=ll)
    
    #Label the plot as required
    plt.grid()
    plt.legend(title='Linking Length')
    plt.ylabel('N + 1')
    plt.xlabel(r'$\beta_M$')
    plt.title(r'Log Mass Growth for halos with $N_p \geq$ 1000 Comparison')
    plt.savefig("report_plots/log_mass_growth/lmg_mbl_comp.png")
    


log_mass_growth()


        






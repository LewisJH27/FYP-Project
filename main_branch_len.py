import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import h5py
import time
import pickle
#master main branch len section -----------------------------------------------------------------------------------
file0 = h5py.File('data/dgraph0.2/Mgraph_0098.hdf5','r')
#print(file.keys())
size = len(file0['halo_IDs'])
z0_mass = file0['nparts']
n = '98'



halo = file0['halo_IDs'][0]

def mainbranchlen():
    lengths = np.zeros(size, dtype=int)
    for ind, halo in enumerate(file0['halo_IDs']):
        print(halo)
        length = 0
        for n in range(98, -1, -1):
            if n < 10:
                file = h5py.File('data/dgraph0.2/Mgraph_000'+str(n)+'.hdf5','r')
                
            else:
                
                file = h5py.File('data/dgraph0.2/Mgraph_00'+str(n)+'.hdf5','r')

            #print(len(file['halo_IDs'][:]))
            #print(type(halo))
            #prog_mass_con = file['Prog_Mass_Contribution']
            

            nprogs = file['nProgs']
            #print(len(nprogs))
            prog_start_index = file['prog_start_index']

            progids = file['Prog_haloIDs']

            nprog = nprogs[halo]

            if nprog == 0 or nprog == -2:
                break
            

            start = prog_start_index[halo]

            # num_of_halos = []
            # for i in range(file["prog_start_index"].size):
            #     if nprogs[i] > 0:
            #         num_of_halos.append(i)
            #         print(i, nprogs[i], prog_start_index[i], progids[prog_start_index[i] : prog_start_index[i] + nprogs[i]])

            # print(len(num_of_halos))
            # print(prog_mass_con[368])



            link_data = progids[prog_start_index[halo] : prog_start_index[halo] + nprogs[halo]]
            length += 1
            #print(int(link_data))
            prog = int(link_data[0])
            halo = prog

            file.close()



        
        lengths[ind] = length
        

    print(lengths)
    return lengths
#---------------------------------------------------------------------------------------------










# hist_data1 = mainbranchlen()
# hist_data = []
# for x in hist_data1:
#     if x != 0:
#         hist_data.append(x)

# plt.figure(0)
# plt.xlabel(r'$\ell$')
# plt.ylabel('$N$')
# hist1, base = np.histogram(hist_data, bins=bins_array)
# cs_data = np.cumsum(hist1)
# plt.plot(base[:-1],cs_data, label='0.16')
# cs_data = np.cumsum(hist_data)
# plt.plot(cs_data,bins_array)
# plt.hist(cs_data, bins=bins_array, label = '0.16')


# file0.close()
#HISTOGRAM PLOT STUFF ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


file1 = h5py.File('data/dgraph0.25/Mgraph_0098.hdf5','r')
#print(file.keys())
size = len(file1['halo_IDs'])
z0_mass = file1['nparts']
n = '98'



halo = file1['halo_IDs'][0]

def mainbranchlen1():
    lengths = np.zeros(size, dtype=int)
    for ind, halo in enumerate(file1['halo_IDs']):
        print(halo)
        length = 0
        for n in range(98, -1, -1):
            if n < 10:
                file2 = h5py.File('data/dgraph0.25/Mgraph_000'+str(n)+'.hdf5','r')
                
            else:
                
                file2 = h5py.File('data/dgraph0.25/Mgraph_00'+str(n)+'.hdf5','r')

            #print(len(file['halo_IDs'][:]))
            #print(type(halo))
            #prog_mass_con = file['Prog_Mass_Contribution']
            

            nprogs = file2['nProgs']
            #print(len(nprogs))
            prog_start_index = file2['prog_start_index']

            progids = file2['Prog_haloIDs']

            nprog = nprogs[halo]

            if nprog == 0 or nprog == -2:
                break
            

            start = prog_start_index[halo]

            # num_of_halos = []
            # for i in range(file["prog_start_index"].size):
            #     if nprogs[i] > 0:
            #         num_of_halos.append(i)
            #         print(i, nprogs[i], prog_start_index[i], progids[prog_start_index[i] : prog_start_index[i] + nprogs[i]])

            # print(len(num_of_halos))
            # print(prog_mass_con[368])



            link_data = progids[prog_start_index[halo] : prog_start_index[halo] + nprogs[halo]]
            length += 1
            #print(int(link_data))
            prog = int(link_data[0])
            halo = prog

            file2.close()



        
        lengths[ind] = length
        

    print(lengths)
    return lengths

bins_array = []
for i in range(0, 98):
    bins_array.append(i)
nparts = file1['nparts']
lengths = mainbranchlen1()
low_threshs = np.ndarray([0,10,100])
up_threshs = np.ndarray([10,100,1000])
outpath = 'report_plots/main_len_plot/'
lims={}
# Set up figure
fig = plt.figure()
gs = gridspec.GridSpec(nrows=3, ncols=1)
gs.update(wspace=0.5, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

for ax, low, up in zip([ax3, ax2, ax1], low_threshs, up_threshs):

    okinds = np.logical_and(nparts >= low, nparts < up)
    ls = lengths[okinds]

    H, _ = np.histogram(ls, bins=bins_array)

    ax.bar(bins_array[:-1] + 0.5, H, width=1, alpha=0.8,
           color="r", edgecolor="r")

    lims[up] = H.max()

# Label axes
ax3.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$N$')

# Annotate the mass bins
ax1.text(0.05, 0.8, r'$M_{H}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
ax2.text(0.05, 0.8, r'$1000\geq M_{H}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
ax3.text(0.05, 0.8, r'$100\geq M_{H}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

# Remove x axis from upper subplots
ax1.tick_params(axis='x', bottom=False, left=False)
ax2.tick_params(axis='x', bottom=False, left=False)

# Set y axis limits such that 0 is removed from the upper two subplots to avoid tick stacking
ax1.set_ylim(0.5, None)
ax2.set_ylim(0.5, None)

# Save figure with a transparent background
fig.savefig(outpath + 'mainbranchlengthcomp.png', bbox_inches="tight")



# bins_array = []
# for i in range(0, 98):
#     bins_array.append(i)


# hist_data3 = mainbranchlen1()
# hist_data2 = []
# for x in hist_data3:
#     if x != 0:
#         hist_data2.append(x)





# cs_data2 = np.cumsum(hist_data2)
# plt.plot(cs_data2, bins_array)
# plt.hist(cs_data2, bins=bins_array, color='r', label = '0.25')

# hist2, base = np.histogram(hist_data2, bins=bins_array)
# cs_data2 = np.cumsum(hist2)
# plt.plot(base[:-1],cs_data2, label='0.25',color='r')

# plt.legend(title='linking length')
# plt.title('Main Branch Length Comparison - Cumulative Sum of Histogram')
# plt.savefig('report_plots/comparison_plots/main_branch_len_hist_cs.png')
# plt.show()


    

  

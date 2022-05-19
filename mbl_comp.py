import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
import pickle
import sys
import h5py
from scipy.stats import ks_2samp
import pickle as pkl
import astropy
from astropy.cosmology import WMAP9 as cosmo

def mainbranchlengthDMLJ(datapath, basename, snaplist):
    """ A function that walks all z=0 halos' main branches computing the main branch length for each.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: lengths: An array of main branch lengths for all halos above the cutoff.
    """

    # Open z=0 direct graph file
    hdf = h5py.File(datapath + basename + snaplist[-1] + ".hdf5", "r")
    size = len(hdf['halo_IDs'])

    # Initialise the lengths array
    lengths = np.zeros(size, dtype=int)

    # Get the z=0 halo ids and masses
    z0halos = hdf['halo_IDs'][:]
    z0nparts = hdf['nparts'][:]

    hdf.close()

    # Loop over the z=0 halos walking down the main branch
    for z0halo in z0halos:

        #print(z0halo, "of", size, end="\r")

        # Initialise the length counter, nprog, halo pointer and snapshot
        length = 0
        nprog = 100
        halo = z0halo
        snapshot = str(int(snaplist[-1]))

        # Loop until a halo with no progenitors is found
        while nprog > 0:

            # Open current snapshot
            hdf = h5py.File(datapath + basename + snapshot.zfill(4) + ".hdf5", "r")

            # Extract number of progenitors and start index
            nprog = hdf["nProgs"][halo]
            if nprog > 0:
                prog_start = hdf["prog_start_index"][halo]

                # Extract the main progenitor
                prog = hdf["Prog_haloIDs"][prog_start]

                # Move halo pointer to progenitor
                halo = prog

                # Compute the progenitor snapshot ID
                snapshot = str(int(snapshot) - 1)

                # Increment the length counter
                length += 1

            hdf.close()

        # Assign the main branch length to the lengths array
        lengths[z0halo] = length

    return lengths, z0nparts


def mainBranchLengthCompPlot():
    """ A function which walks the main branches of any algorithms with data in the supplied directory with the
    correct format (during this project this was SMT comparison project algorithms) and the main branches produced
    by DMLumberJack, produces histograms in 3 mass bins (20<=M<100, 100<=M<1000, 100<=M) and produces
    a plot of all 3 bins comparing algorithms.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param SMTtreepath: The file path for the SMT algorithm data files.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: None
    """

    # Get commandline paths
    datapath = sys.argv[1]
    datapath2 = sys.argv[2]
    basename = sys.argv[3]
    snapfile = sys.argv[4]
    outpath = sys.argv[5]


    # Load the snapshot list
    snaplist = list(np.loadtxt(snapfile, dtype=str))

    # Create lists of lower and upper mass thresholds for histograms
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    # Compute DMLumberJack's main branch lengths
    # lengths, nparts = mainbranchlengthDMLJ(datapath, basename, snaplist)
    # lengths1, nparts1 = mainbranchlengthDMLJ(datapath2, basename, snaplist)
    infile = open('report_plots/main_len_plot/mbl_0.16','rb')
    dataset_16 = pkl.load(infile)
    infile.close()

    infile1 = open('report_plots/main_len_plot/mbl_0.25','rb')
    dataset_25 = pkl.load(infile1)
    infile1.close()


    lengths = dataset_16[0]
    lengths1 = dataset_25[0]

    nparts = dataset_16[1]
    nparts1= dataset_25[1]
    print(nparts.max())
    bin_edges = np.arange(0, int(snaplist[-1]), 1)

    lims = {}

    # data_sets = [lengths, lengths1]
    # =============== Plot the results ===============
    colours = ['r','b']
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

        H, _ = np.histogram(ls, bins=bin_edges)
        H_cs = np.cumsum(H)
        ax.plot(bin_edges[:-1] + 0.5, H_cs,
            color='b', label='0.16')
        #Show legend
        ax.legend(title='Linking Length', loc='upper center', prop=dict(size=8), title_fontsize='x-small')
        
        # ax.bar(bin_edges[:-1] + 0.5, H, width=1, alpha=0.8,
        #        color="r", edgecolor="r")

        lims[up] = H.max()

        okinds1 = np.logical_and(nparts1 >= low, nparts1 < up)
        ls1 = lengths1[okinds1]

        H1, _ = np.histogram(ls1, bins=bin_edges)
        H_cs1 = np.cumsum(H1)
        ax.plot(bin_edges[:-1] + 0.5, H_cs1,
            color='r',label='0.25')
        #Show legend
        ax.legend(title='Linking Length', loc='upper center', prop=dict(size=8), title_fontsize='x-small')
        
        # ax.bar(bin_edges[:-1] + 0.5, H, width=1, alpha=0.8,
        #        color="r", edgecolor="r")

        lims[up] = H1.max()
        
        


    

    
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

    plt.annotate(r'$M_{H}>1000$'+': KS - S = 0.1134020618155670, P = 0.5632950213999474', (0,0), (-20, -40), xycoords='axes fraction',
                textcoords='offset points', va='top')
    plt.annotate(r'$1000\geq M_{H}>100$'+': KS - S = 0.14432989690721648, P = 0.2654793849715836', (0,0), (-20, -60), xycoords='axes fraction',
                textcoords='offset points', va='top')
    plt.annotate(r'$100\geq M_{H}>20$'+': KS - S = 0.21649484536082475, P = 0.0209519039352964', (0,0), (-20, -80), xycoords='axes fraction',
                textcoords='offset points', va='top')
    
    
    #Set figure title
    fig.suptitle('Main Branch Length histogram - Cumulative sum: linking lengths 0.16 & 0.25')

    # Save figure with a transparent background
    fig.savefig(outpath + 'mblplot_comp.png', bbox_inches="tight")


# mainBranchLengthCompPlot()


def mainBranchLengthNormCompPlot():
    """ A function which walks the main branches of any algorithms with data in the supplied directory with the
    correct format (during this project this was SMT comparison project algorithms) and the main branches produced
    by DMLumberJack, produces histograms in 3 mass bins (20<=M<100, 100<=M<1000, 100<=M) and produces
    a plot of all 3 bins comparing algorithms.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param SMTtreepath: The file path for the SMT algorithm data files.
    :param cutoff: The halo mass cutoff in number of particles. Halos under this mass threshold are skipped.

    :return: None
    """

    # Get commandline paths
    # datapath = sys.argv[1]
    # datapath2 = sys.argv[2]
    # basename = sys.argv[3]
    # snapfile = sys.argv[4]
    # outpath = sys.argv[5]

    datapath = sys.argv[1]
    basename = sys.argv[2]
    snapfile = sys.argv[3]
    outpath = sys.argv[4]
    # Load the snapshot list
    snaplist = list(np.loadtxt(snapfile, dtype=str))

    # Create lists of lower and upper mass thresholds for histograms
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    # Compute DMLumberJack's main branch lengths
    lengths3, nparts3 = mainbranchlengthDMLJ(datapath, basename, snaplist)
    # lengths, nparts = mainbranchlengthDMLJ(datapath2, basename, snaplist)
    infile = open('report_plots/main_len_plot/mbl_0.2','rb')
    dataset_02 = pkl.load(infile)
    infile.close()

    infile1 = open('report_plots/main_len_plot/mbl_0.5','rb')
    dataset_05 = pkl.load(infile1)
    infile1.close()


    lengths = dataset_02[0]
    lengths1 = dataset_05[0]

    nparts = dataset_02[1]
    nparts1= dataset_05[1]
    bin_edges = np.arange(0, int(snaplist[-1]), 1)

    lims = {}

    # data_sets = [lengths, lengths1]
    # =============== Plot the results ===============
    colours = ['r','b']
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

        H, _ = np.histogram(ls, bins=bin_edges)
        H_cs = np.cumsum(H)
        ax.plot(bin_edges[:-1] + 0.5, H_cs/H_cs[-1],
            color='b', label='0.2')
        #Show legend
        ax.legend(title='Linking Length', loc='upper center', prop=dict(size=6), title_fontsize='x-small')
        
        # ax.bar(bin_edges[:-1] + 0.5, H, width=1, alpha=0.8,
        #        color="r", edgecolor="r")

        lims[up] = H.max()

        
        
        okinds2 = np.logical_and(nparts3 >= low, nparts3 < up)
        ls3 = lengths3[okinds2]

        H3, _ = np.histogram(ls3, bins=bin_edges)
        H_cs3 = np.cumsum(H3)
        ax.plot(bin_edges[:-1] + 0.5, H_cs3/H_cs3[-1],
            color='g',label='0.4')
        #Show legend
        ax.legend(title='Linking Length', loc='upper center', prop=dict(size=6), title_fontsize='x-small')
        
        # ax.bar(bin_edges[:-1] + 0.5, H, width=1, alpha=0.8,
        #        color="r", edgecolor="r")

        lims[up] = H3.max()


        okinds1 = np.logical_and(nparts1 >= low, nparts1 < up)
        ls1 = lengths1[okinds1]

        H1, _ = np.histogram(ls1, bins=bin_edges)
        H_cs1 = np.cumsum(H1)
        ax.plot(bin_edges[:-1] + 0.5, H_cs1/H_cs1[-1],
            color='r',label='0.5')
        #Show legend
        ax.legend(title='Linking Length', loc='upper center', prop=dict(size=6), title_fontsize='x-small')
        
        # ax.bar(bin_edges[:-1] + 0.5, H, width=1, alpha=0.8,
        #        color="r", edgecolor="r")

        lims[up] = H1.max()


    

    
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
    ax1.tick_params(axis='x', bottom=False, left=False, right=False)
    ax2.tick_params(axis='x', bottom=False, left=False, right=False)

    # Set y axis limits such that 0 is removed from the upper two subplots to avoid tick stacking
    ax1.set_ylim(0.001, None)
    ax2.set_ylim(0.001, None)

    # plt.annotate(r'$M_{H}>1000$'+': KS - S = 0.1134020618155670, P = 0.5632950213999474', (0,0), (-20, -40), xycoords='axes fraction',
    #             textcoords='offset points', va='top')
    # plt.annotate(r'$1000\geq M_{H}>100$'+': KS - S = 0.14432989690721648, P = 0.2654793849715836', (0,0), (-20, -60), xycoords='axes fraction',
    #             textcoords='offset points', va='top')
    # plt.annotate(r'$100\geq M_{H}>20$'+': KS - S = 0.21649484536082475, P = 0.0209519039352964', (0,0), (-20, -80), xycoords='axes fraction',
    #             textcoords='offset points', va='top')
    
    
    #Set figure title
    fig.suptitle('Main Branch Length histogram - Cumulative sum: linking lengths 0.2, 0.4, 0.5')

    # Save figure with a transparent background
    fig.savefig(outpath + 'mblplot_comp_normsed_large-lls.png', bbox_inches="tight")


mainBranchLengthNormCompPlot()





def pickle_mbl():
    
    datapath = sys.argv[1]
    # datapath2 = sys.argv[2]
    basename = sys.argv[2]
    snapfile = sys.argv[3]
    outpath = sys.argv[4]

    snaplist = list(np.loadtxt(snapfile, dtype=str))

    lengths, nparts = mainbranchlengthDMLJ(datapath, basename, snaplist)
    # lengths1, nparts1 = mainbranchlengthDMLJ(datapath2, basename, snaplist)
    dataset_02 = [lengths, nparts]
    # dataset_02 = [lengths1, nparts1]
    # filename = 'report_plots/main_len_plot/mbl_0.4'

    # outfile = open(filename,'wb')
    # pkl.dump(dataset_02,outfile)
    # outfile.close()

    filename1 = 'report_plots/main_len_plot/mbl_0.2'

    outfile1 = open(filename1,'wb')
    pkl.dump(dataset_02,outfile1)
    outfile1.close()
# pickle_mbl()

def ks_test_mbl():

    # Get commandline paths
    datapath = sys.argv[1]
    datapath2 = sys.argv[2]
    basename = sys.argv[3]
    snapfile = sys.argv[4]
    outpath = sys.argv[5]


    # Load the snapshot list
    snaplist = list(np.loadtxt(snapfile, dtype=str))


    #unpickle the data
    infile = open('report_plots/main_len_plot/mbl_0.16','rb')
    dataset_16 = pkl.load(infile)
    infile.close()

    #unpickle the data
    infile1 = open('report_plots/main_len_plot/mbl_0.25','rb')
    dataset_25 = pkl.load(infile1)
    infile1.close()

    #Setup the mass thresholds 
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    #assign lengths and nparts to variables
    lengths = dataset_16[0]
    lengths1 = dataset_25[0]

    nparts = dataset_16[1]
    nparts1= dataset_25[1]
    
    bin_edges = np.arange(0, int(snaplist[-1]), 1)

    ks = []
    
    for low, up in zip(low_threshs, up_threshs):
        okinds = np.logical_and(nparts >= low, nparts < up)
        ls = lengths[okinds]
        
        H, _ = np.histogram(ls, bins=bin_edges)
        H_cs = np.cumsum(H)
        print('H=',H)
        print('finished')

        

        okinds1 = np.logical_and(nparts1 >= low, nparts1 < up)
        ls1 = lengths1[okinds1]

        H1, _ = np.histogram(ls1, bins=bin_edges)
        H_cs1 = np.cumsum(H1)

        ks.append(ks_2samp(H_cs,H_cs1))
        
    # print(ks)
    
# ks_test_mbl()


#KstestResult(statistic=0.21649484536082475, pvalue=0.0209519039352964),
#KstestResult(statistic=0.14432989690721648, pvalue=0.2654793849715836)
#KstestResult(statistic=0.1134020618155670, pvalue=0.5632950213999474)
def get_times():
    redshift_array = []

    for i in range(0, 99):
            
            snapshot_num = str(i)
            if len(snapshot_num) == 1:
                file_number = '0' + snapshot_num
                #print(file_number
            if len(snapshot_num) == 2:
                file_number = snapshot_num
                #print(file_number)
            
            f = h5py.File('data/halos-0.4/halos_00'+file_number+'.hdf5','r')

            red_shift = f.attrs['redshift']
            redshift_array.append(red_shift)
        #Define an array for the times and then calculate them using astropy    
    cosmo_time = []

        
    for n in redshift_array:

        cosmo_time.append((cosmo.age(n)).value[0])


    dicts = {}
    snaplist = list(np.loadtxt('snaplist3.txt', dtype=str))

    for i in range(len(cosmo_time)):
        dicts[snaplist[i]] = cosmo_time[i]

    return dicts

############HOW TO RUN #############################
#mpirun -np 4 python mbl_comp.py data/dgraph0.16/ data/dgraph0.25/ Mgraph_ snaplist2.txt report_plots/main_len_plot/


    




import h5py
import statistics
import numpy as np
import astropy
from astropy.cosmology import WMAP9 as cosmo



redshift_array = []

for i in range(0, 99):
    
    snapshot_num = str(i)
    if len(snapshot_num) == 1:
        file_number = '0' + snapshot_num
        #print(file_number
    if len(snapshot_num) == 2:
        file_number = snapshot_num
        #print(file_number)

    f = h5py.File('data/halos-0.2/halos_00'+file_number+'.hdf5','r')

    red_shift = f.attrs['redshift']
    redshift_array.append(red_shift)
    
cosmo_time = []

for n in redshift_array:
    cosmo_time.append((cosmo.age(n)).value[0])

print(cosmo_time)







import ctypes
import rebound
import numpy as np
import tools
import pandas as pd
sim = rebound.Simulationarchive('../data/Single/156/archive.bin')
fullfile='../data/Single/156/archive.bin'
flag2, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(sbody = str('156'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
h_init = e_init*np.sin(aop_init+lan_init)
k_init = e_init*np.cos(aop_init+lan_init)
print('read file')
import ctypes
from numpy.ctypeslib import ndpointer 

fmft = ctypes.cdll.LoadLibrary('./fmft.so')

#_foobar = fmft.foobar 
#_foobar.argtypes = [ctypes.c_int, ctypes.c_int, _doublepp, _doublepp] 
#_foobar.restype = None 

fmft = ctypes.cdll.LoadLibrary('./fmft.so')
n = len(h_init)

flag = 2
nfreq=4
fmft.fmft.argtypes = [ctypes.c_int,
                      np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,shape=(3*flag+1,nfreq)), 
                      ctypes.c_int, 
                      ctypes.c_double, 
                      ctypes.c_double, 
                      ctypes.c_int, 
                      np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,shape=(3,n)), 
                      ctypes.c_long]
fmft.fmft.restype = None

DATA_SEP=sim[1].t
maxfreq=ctypes.c_double(1/180./3600. * np.pi *DATA_SEP)
minfreq=ctypes.c_double(0.01/180./3600. * np.pi *DATA_SEP)
maxfreq=ctypes.c_double(1/1000)
minfreq=ctypes.c_double(1/10000000)
#flag = 2
ndata=8192
# Create input and output numpy arrays
print('converting')
input_array = np.array([t_init,k_init,h_init])
output_array = np.zeros((3*flag+1, nfreq))

# Convert numpy arrays to ctypes pointers
#input_ptr = (ctypes.POINTER(ctypes.c_double))()
#output_ptr = (ctypes.POINTER(ctypes.c_double))()

#for i in range(len(input_array)):
#    input_ptr[i] = input_array[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#for i in range(len(output_array)):
#    output_ptr[i] = output_array[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Call the C function
print(len(input_array),len(input_array[0]))

print('running fmft')
outs= fmft.fmft(n,output_array, nfreq, minfreq, maxfreq, flag, input_array, ndata)
print(output_array)
# Print the output array
for i in range(len(output_array[0])):
    print("Frequency:",output_array[3,i]*180.*3600./np.pi/sim[1].t)
    print("Amplitude:",output_array[4,i])
    print("Phase:",output_array[5,i])



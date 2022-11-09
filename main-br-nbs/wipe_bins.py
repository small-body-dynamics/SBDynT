import numpy as np
import pandas as pd
import sys
import os


astdys = pd.read_csv('astdys_tnos.csv')

#for i in range(len(astdys)):
#for i in range(10):
objnum = int(sys.argv[1])
#print(objnum)
objname = astdys['Name'].iloc[objnum]
filename = 'TNOs/' + objname

if os.path.isfile(filename+'/archive.bin'):
    os.remove(filename+'/archive.bin')

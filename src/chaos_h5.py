import os
import numpy as np
import pandas as pd
import h5py
import chaos_indicators

file = '../data/CC_chaos_map/'

filenames = os.listdir(file)
filtered_names = [s for s in filenames if 'init' not in s]



print(filtered_names)
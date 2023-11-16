import os
import pandas as pd
from datetime import datetime

from schwimmbad import MultiPool

astdys = pd.read_csv('data_files/AstFam_data.csv')

begin = datetime.now()
def run(i):
    string = 'python make_f.py AstFam '+str(i)+' 8planet'
    os.system(string)
    return None

data = range(1800,len(astdys))
with MultiPool() as pool:
    values = list(pool.map(run, data))
    
    print(datetime.now() - begin)

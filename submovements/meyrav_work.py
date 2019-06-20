from DataProcessing import Preprocessor, Trial, Subject
import pandas as pd
import numpy as np
import os
import re
import attr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, hilbert
import seaborn as sns; sns.set()

if __name__ == "__main__":
    #pproc = Preproces
    #trial_gen = pproc.load_df_from_directory_gen('../1')
    #for trial in trial_gen
    a = Subject('../1', r'C:\Users\meyra\Desktop\s')
    a.create_total_df()
    #print(a.df_dict['tri_left'])
    #print(a.df_total)
#    idx = pd.IndexSlice
#    for block in range(int(a.max_block_dict['tri_left'])):
#        for rep in range(int(a.max_rep_dict['tri_left'])):
#            z = a.df_total.loc[idx[a.id,'tri_left',block,rep,:],:]
#    print(z)
    a.stimuli_plot(10)
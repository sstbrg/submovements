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
    a = Subject('../1', r'C:\Users\meyra\Desktop\s')
    a.create_total_df()
    a.stimuli_plot_vx(10)
    a.stimuli_plot_vy(10)

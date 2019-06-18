# -*- coding: utf-8 -*-

from DataProcessing import Preprocessor

path_to_raw_data = '..\\data\\results\\simonvisual\\1'
path_to_database = '..\\data\\processed_data.mat'
sample_rate = 240 #S/s

pproc = Preprocessor(sample_rate=sample_rate)
raw_data_gen = pproc.load_df_from_directory_gen(dir_path=path_to_raw_data)

for ii, file_name in raw_data_gen:
    print(file_name)
    print(ii)


from DataProcessing import Preprocessor

if __name__ == '__main__':
    path_to_raw_data = '..\\data\\results\\simonvisual\\1'
    path_to_database = '..\\data\\processed_data.mat'
    path_to_trials = '..\\data\\trials'

    pproc = Preprocessor()
    raw_data_gen = pproc.load_df_from_directory_gen(dir_path=path_to_raw_data, cols=('x', 'y'))

    for trial in raw_data_gen:
        print(trial.raw_file_path)
        trial.preprocess(pproc)
        trial.save_df(trial.data, path_to_trials)

        #pproc.plot(trial.data[['Vx', 'Vy']])
        #pproc.plot(trial.data[['x', 'y']])
        #break

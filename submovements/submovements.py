from DataProcessing import Preprocessor

if __name__ == '__main__':
    path_to_raw_data = '..\\data\\results\\simonvisual\\1'
    path_to_database = '..\\data\\processed_data.mat'
    path_to_trials = '..\\data\\trials'

    pproc = Preprocessor()
    raw_data_gen = pproc.load_df_from_directory_gen(dir_path=path_to_raw_data)

    for trial in raw_data_gen:
        trial.preprocess(pproc, axes=['x', 'y'])
        trial.save_as_csv(path_to_trials)
        pproc.plot(trial.velocity_data[['x', 'y']])
        pproc.plot(trial.filtered_velocity_data[['x', 'y']])
        pproc.plot(trial.filtered_position_data[['x', 'y']])
        break

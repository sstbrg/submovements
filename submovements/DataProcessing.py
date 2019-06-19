import attr
from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import butter, filtfilt

@attr.s
class Trial(object):

    ###
    # Trial(i,j) represents data from repetition i from block j
    # Using the preprocessor we can stream data from a directory and create trials
    # Using the preprocessor we can preprocess the trial position data into filtered velocity
    ###

    block = attr.ib(default=0)
    rep = attr.ib(default=0)
    stimulus = attr.ib(default='')
    position_data = attr.ib(default=None)
    velocity_data = attr.ib(default=None)
    filtered_velocity_data = attr.ib(default=None)
    events = attr.ib(default=None)
    raw_file_path = attr.ib(default='')

    def __str__(self):
        return f'Block/Repetition: {self.block}/{self.rep}\n' \
            f'Stimulus: {self.stimulus}\n' \
            f'Raw data path: {self.raw_file_path}\n'

    def preprocess(self, preprocessor, axes=('x','y','z'), sg_win_len = 17, sg_polyorder=4, threshold=0.005):
        filtered_velocity = preprocessor.filter_raw_data(self.position_data[list(axes)],
                                                  window_len=sg_win_len,
                                                  polyorder=sg_polyorder,
                                                  deriv=1)
        filtered_velocity = preprocessor.remove_baseline(filtered_velocity, threshold=threshold)

        self.filtered_velocity_data = filtered_velocity

    def save_as_csv(self, dest_folder):
        ### TODO
        # Method to export a Trial to csv
        ###

        df = self.filtered_velocity_data.copy()
        df.columns = ['Vx', 'Vy']
        filename = f"li_{self.stimulus}_{self.block}_{self.rep}.csv"
        filepath = os.path.join(dest_folder, filename)

        df.to_csv(filepath)

@attr.s
class Preprocessor(object):
    raw_paths = attr.ib(default='data')
    database_file_path = attr.ib(default='data\\database.mat')
    sample_rate = attr.ib(default=240)
    raw_headers = attr.ib(default=['SampleNum', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'Time', 'Event'])

    def load_df_from_directory_gen(self, dir_path):
        ###
        # This is a generator which takes csv files from dir_path and yields Trials.
        ###

        self.raw_paths = glob(os.path.join(dir_path, 'li_*.csv'))
        trial_out = Trial()
        for fn in self.raw_paths:
            try:
                df = pd.read_csv(fn, names=self.raw_headers)
                df = df.set_index('Time')
                trial_out.position_data = df[['x','y','z']]
                trial_out.events = df[['Event']]
                trial_out.velocity_data = df[['x','y','z']].diff()
                trial_data = os.path.split(fn)[1].split(sep='_')
                trial_out.stimulus = trial_data[1] + '_' + trial_data[2]
                trial_out.block = int(trial_data[3])
                trial_out.rep = int(os.path.splitext(trial_data[4])[0])
                trial_out.raw_file_path = fn
                yield trial_out

            except IndexError:
                print(f'File does not contain trial data: {fn}')
                continue

            except IOError:
                raise AssertionError(f'Could not load {fn}.')


    def load_single_file(self, file_path):
        ###
        # This method loads a single csv and yields a single Trial
        ###

        trial_out = Trial()
        try:
            trial_out.data = pd.read_csv(file_path, names=self.raw_headers)
            trial_data = os.path.split(file_path)[1].split(sep='_')
            trial_out.subject_id = trial_data[0]
            trial_out.stimulus = trial_data[1]
            trial_out.raw_file_path = file_path
            return trial_out
        except IOError:
            raise AssertionError(f'{file_path} was not loaded.')

    @staticmethod
    def plot(data_in: pd.DataFrame, sample_factor: int = 10):

        ###
        # This is a plotting method that will work on a given DataFrame
        # mode = 'pandas' will plot using Pandas and matplotlib.
        # mode = 'bokeh' will plot using bokeh.
        # sample_factor dictates how many samples will be displayed (lower -> higher performance -> lower resolution).
        ###

        plt.figure()
        data_in.plot()
        plt.show()

    @staticmethod
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=10):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def filter_raw_data(self, data_in:pd.DataFrame, window_len=5, deriv=1, polyorder=2):
        ###
        # This method applies the Savitsky-Golay filter on a data frame.
        # For example, the velocity is extracted by setting deriv=1 and data_in to position data.
        #
        # S-G paramters:
        # window_len - how many samples are used for polynomial fitting of order polyorder.
        # deriv=n controls whether we smooth the data or its n'th derivative.
        ###

        df = data_in.copy()
        dx = df.index[1]

        # we start by filling NaNs in the data
        df = df.fillna(method='bfill')

        # 5hz low pass (zero phase)
        cutoff = 5 #hz
        order = 10

        for col in df:
            df[col] = self.butter_lowpass_filter(df[col], cutoff=cutoff, fs=self.sample_rate, order=order)


        # we now apply a Savitzky-Golay filter to smooth the data
        for col in df:
            df[col] = savgol_filter(x=df[col],
                                    window_length=window_len,
                                    polyorder=polyorder,
                                    deriv=deriv,
                                    delta=dx)
        return df

    @staticmethod
    def remove_baseline(data_in:pd.DataFrame, threshold=0.005):
        ### TODO: check if we need to filter out baseline data --after-- the motion...
        # This method takes a data frame of velocity data, calculates the normalized magnitude of
        # tangential velocity and filters out any data that's lower than a given threshold.
        ###

        # calculate absolute velocity
        df = np.power(data_in, 2)
        df = df.sum(axis=1)

        # min-max normalization
        df = (df - df.min()) / (df.max() - df.min())

        # find data above threshold
        idx = df.loc[df >= threshold]

        # set data cutting limits
        low_cut_index = idx.index.min()-0.2 if df.index.min() < idx.index.min()-0.2 else df.index.min()
        high_cut_index = idx.index.max()+0.2 if df.index.max() > idx.index.max()+0.2 else df.index.max()

        return data_in.copy()[low_cut_index : high_cut_index]


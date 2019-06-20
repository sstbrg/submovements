import attr
from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import re


@attr.s
class Trial():
    ###
    # Trial(i,j) represents data from repetition i and block j
    # Using the preprocessor we can stream data from a
    # directory and create trials.
    # Using the preprocessor we can preprocess the trial
    # position data into filtered velocity
    ###

    block = attr.ib(default=0)          # Trial block
    rep = attr.ib(default=0)            # Trial repetition
    stimulus = attr.ib(default='')      # Trial stimulus
    raw_file_path = attr.ib(default='') # Path to trial raw data
    id = attr.ib(default='')            # Subject ID (number)
    time = attr.ib(default='')          # Time vector

    # Data DataFrame which contains Position vectors and Velocity vectors.
    # We can choose which vectors to include from ('x','y','z') by using
    # the cols parameter. For example, cols=('x','y') will choose only dimensions x and y.

    data = attr.ib(default='')

    def preprocess(self, preprocessor,
                   cols=('x', 'y'),
                   threshold=0.05):
        ###
        # This method does Trial preprocessing using a Preprocessor
        # in directions chosen by cols.
        # Threshold refers to baseline removal. Data where ||Velocity|| < threshold
        # is removed. Threshold is given in percentage w.r.t to max(||Velocity||).
        ###

        assert isinstance(preprocessor, Preprocessor)

        cols=list(cols)
        self.data[cols] = preprocessor.filter_raw_data(self.data[cols])

        velocity_cols = [f'V{q}' for q in cols]
        self.data[velocity_cols] = preprocessor.sample_rate * \
                                             self.data[cols].diff().fillna(method='bfill')

        self.data = preprocessor.remove_baseline(self.data, cols=velocity_cols, threshold=threshold)

        self.data = self.data.set_index(self.data['Time']-self.data['Time'][0])

    # def save_as_csv(self, dest_folder):
    #     ###
    #     # This method saves a Trial into a CSV
    #     ###
    #
    #     assert Path(dest_folder).is_dir(), \
    #         f'Destination directory does not exists: {dest_folder}'
    #
    #     dest_folder = Path(dest_folder)
    #     df = self.data.copy()
    #     filename = f"li_{self.stimulus}_{self.block}_{self.rep}.csv"
    #     filepath = dest_folder.joinpath(filename)
    #
    #     df.to_csv(filepath)

    def create_df(self):
        ###
        # Creates a DataFrame for every trial with the columns:
        # Vx, Vy, Condition, Time, ID, Block, Repetition
        # TODO: Generalize to z axis
        ###

        vx = self.data['Vx']
        vy = self.data['Vy']
        time = np.arange(len(vx))    # change later !!!!
        condition = np.full(shape=len(vx), fill_value=self.stimulus)

        # maybe add to main as trial attribute
        id = np.full(shape=len(vx),
                     fill_value=self.id
                     )
        block = np.full(shape=len(vx), fill_value=self.block, dtype=np.int)
        rep = np.full(shape=len(vx), fill_value=self.rep, dtype=np.int)
        return pd.DataFrame({'Vx': vx, 'Vy': vy, 'Rep': rep, 'Block': block,
                             'Time': time, 'Condition': condition, 'ID': id})

    def save_df(self, df, dest_folder):
        ###
        # Save recived data frame of the trial as a '_df.csv'
        ###

        if isinstance(df, pd.DataFrame):
            fname = f"{self.stimulus}_{self.block}_{self.rep}_df.csv"
            path = os.path.join(dest_folder, fname)
            if os.path.isdir(dest_folder):
                df.to_csv(path)
            else:
                try:
                    os.mkdir(dest_folder)
                    df.to_csv(path)
                except OSError:
                    print("Creation of the directory %s failed" % path)


@attr.s
class Preprocessor():
    ###
    # This is a Preprocessing entity which can filter,
    # cut-via-threshold and plot DataFrames
    ###

    raw_paths = attr.ib(default='data') # Path list of all the raw data files
    sample_rate = attr.ib(default=240)  # Sample rate the raw data is sampled at

    raw_headers = attr.ib(default=['SampleNum', 'x', 'y', 'z',
                                   'phi', 'theta', 'psi', 'Time', 'Event'])

    def load_df_from_directory_gen(self, dir_path, cols=('x', 'y')):
        ###
        # This is a generator which takes csv
        # files from dir_path and yields Trials.
        # cols controls which dimensions are saved.
        ###

        assert Path(dir_path).is_dir(), \
            f'Destination directory does not exist ' \
            f'or is not a directory: {dir_path}'

        self.raw_paths = glob(str(Path(dir_path).joinpath('li_*_*_*_*.csv')))
        assert len(self.raw_paths) > 0, f'No source files found!'

        trial_out = Trial()
        trial_out.data = pd.DataFrame()
        cols = list(cols)
        for fn in self.raw_paths:
            try:
                df = pd.read_csv(fn, names=self.raw_headers)
                data = df['Time'].astype('float64')
                data = pd.concat([data, df[cols]], axis=1)
                trial_out.data = data
                trial_data = os.path.split(fn)[1].split(sep='_')
                trial_out.stimulus = trial_data[1] + '_' + trial_data[2]
                trial_out.block = int(trial_data[3])
                trial_out.rep = int(os.path.splitext(trial_data[4])[0])
                trial_out.raw_file_path = fn
                trial_out.id = os.path.split(dir_path)[1]
                yield trial_out
            except ValueError:
                raise AssertionError(f'Could not load {fn}.')

    def load_single_file(self, file_path, cols=('x', 'y')):
        ###
        # This method loads a single csv and yields a single Trial.
        # cols controls which dimensions are saved.
        ###

        assert Path(file_path).is_file(), f'File does not exists: {file_path}'
        file_path = Path(file_path)
        cols=list(cols)
        trial_out = Trial()
        try:
            df = pd.read_csv(file_path, names=self.raw_headers)
            data = df['Time'].astype('float64')
            data = pd.concat([data, df[cols]], axis=1)
            trial_out.data = data
            trial_data = os.path.split(file_path)[1].split(sep='_')
            trial_out.stimulus = trial_data[1] + '_' + trial_data[2]
            trial_out.block = int(trial_data[3])
            trial_out.rep = int(os.path.splitext(trial_data[4])[0])
            trial_out.raw_file_path = file_path
            trial_out.id = os.path.split(file_path)[1]
            return trial_out
        except ValueError:
            raise AssertionError(f'Could not load {file_path}.')

    @staticmethod
    def plot(data_in: pd.DataFrame):
        ###
        # This is a plotting method that will work on a given DataFrame.
        ###

        assert isinstance(data_in, pd.DataFrame)
        assert not data_in.empty, f'No data to plot!'

        plt.figure()
        data_in.plot()
        plt.show()

    @staticmethod
    def butter_lowpass(cutoff, fs, order):
        ###
        # fs - sample rate in S/s
        # order - filter order
        ###

        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        ###
        # cutoff - lowpass cutoff frequency in Hz
        # fs - sample rate in S/s
        # order - filter order.
        # Note: since we're using zero-phase filtering
        # the effective order is 2*order.
        # Here the effective order is 2*2=4 by default.
        ###

        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def filter_raw_data(self, data_in: pd.DataFrame, cutoff=5, order=2):
        ###
        # This method applies a 5hz low pass
        # zero-phase butterworth filter.
        ###

        assert isinstance(data_in, pd.DataFrame)
        assert not data_in.empty, f'No data to process!'

        df = data_in.copy()

        # we start by filling NaNs in the data using nearest neighbor.
        df = df.fillna(method='bfill')

        for col in df:
            df[col] = self.butter_lowpass_filter(df[col],
                                                 cutoff=cutoff,
                                                 fs=self.sample_rate,
                                                 order=order)
        return df

    def remove_baseline(self, data_in: pd.DataFrame, cols=('x', 'y'), threshold=0.05):
        ###
        # This method takes a data frame of velocity data,
        # calculates the normalized magnitude of
        # tangential velocity and filters out any data
        # that's lower than a given threshold.
        ###

        assert isinstance(data_in, pd.DataFrame)
        assert not data_in.empty, f'No data to process!'
        assert threshold > 0, f'Threshold must be greater or equal than zero.'
        assert len(cols) > 0, f'Cannot process if no columns are specified.'

        # calculate absolute velocity
        df = np.power(data_in[cols], 2)
        df = np.sqrt(df[cols].sum(axis=1))

        # min-max normalization
        df = (df - df.min()) / (df.max() - df.min())

        # find data above threshold
        idx = df.loc[df >= threshold]

        # expand data cutting limits
        low_cut_index = int(idx.index[0]-0.1*self.sample_rate \
            if df.index.min() < idx.index[0]-0.1*self.sample_rate \
            else df.index.min())
        high_cut_index = int(idx.index[-1]+0.1*self.sample_rate \
            if df.index.max() > idx.index[-1]+0.1*self.sample_rate \
            else df.index.max())

        return data_in.copy()[low_cut_index:high_cut_index].reset_index(drop=True)

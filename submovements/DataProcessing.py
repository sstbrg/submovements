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
class Subject(object):
    dir_path = attr.ib(default='')
    id = attr.ib(init=False)
    df_folder = attr.ib(default='')
#
    def __attrs_post_init__(self):
        self.id = os.path.split(self.dir_path)[1]

    def create_total_df(self):
        self.df_total = pd.DataFrame({'Vx': [], 'Vy': [], 'Rep': [], 'Block': [], 'Time': [], 'Condition': [],
                                      'ID': [], 'pos x':[], 'pos y':[]})  # creating an empty array for concatination use later
        pproc = Preprocessor()
        trial_gen = pproc.load_df_from_directory_gen(self.dir_path)
        for trial in trial_gen:
            trial.preprocess(pproc)
            df = trial.create_df()
            trial.save_df(df, self.df_folder)
            self.df_total = pd.concat([self.df_total, df])
        self.df_total = self.df_total.set_index(['ID', 'Condition', 'Block', 'Rep', 'Time']).sort_values(
            ['ID', 'Condition', 'Block', 'Rep'], ascending=True)

    def stimuli_df(self,stimuli):
        idx = pd.IndexSlice
        return self.df_total.loc[idx[:,stimuli,:,:,:],:]


@attr.s
class Trial(object):
    ###
    # Trial(i,j) represents data from repetition i and block j
    # Using the preprocessor we can stream data from a
    # directory and create trials.
    # Using the preprocessor we can preprocess the trial
    # position data into filtered velocity
    ###

    block = attr.ib(default=0)
    rep = attr.ib(default=0)
    stimulus = attr.ib(default='')
    position_data = attr.ib(default=None)
    velocity_data = attr.ib(default=None)
    filtered_velocity_data = attr.ib(default=None)
    filtered_position_data = attr.ib(default=None)
    events = attr.ib(default=None)
    raw_file_path = attr.ib(default='')
    id = attr.ib(default='')

    def preprocess(self, preprocessor,
                   axes=('x', 'y', 'z'),
                   threshold=0.005):
        assert isinstance(preprocessor, Preprocessor)

        self.filtered_position_data = preprocessor.filter_raw_data(
            self.position_data[list(axes)])

        self.filtered_velocity_data = \
            self.filtered_position_data.diff().fillna(method='bfill')
        self.filtered_velocity_data *= preprocessor.sample_rate

        self.filtered_velocity_data = preprocessor.remove_baseline(
            self.filtered_velocity_data, threshold=threshold)

        self.filtered_position_data = self.filtered_position_data[
                                      self.filtered_velocity_data.index.min():
                                      self.filtered_velocity_data.index.max()]

    def save_as_csv(self, dest_folder):
        assert Path(dest_folder).is_dir(), \
            f'Destination directory does not exists: {dest_folder}'

        dest_folder = Path(dest_folder)
        df = self.filtered_velocity_data.copy()
        df.columns = ['Vx', 'Vy']
        filename = f"li_{self.stimulus}_{self.block}_{self.rep}.csv"
        filepath = dest_folder.joinpath(filename)

        df.to_csv(filepath)

    def create_df(self):
        ###
        # creates df for every trial with the columns:
        # Vx, Vy, Condition, Time, ID, Block, Repetition
        ###

        vx = self.filtered_velocity_data['x']
        vy = self.filtered_velocity_data['y']
        time = np.arange(len(vx))    # change later !!!!
        condition = np.full(shape=len(vx), fill_value=self.stimulus)
        id = np.full(shape=len(vx),
                     fill_value=self.id
                     )
        block = np.full(shape=len(vx), fill_value=self.block, dtype=np.int)
        rep = np.full(shape=len(vx), fill_value=self.rep, dtype=np.int)
        pos_x = self.filtered_position_data['x']
        pos_y = self.filtered_position_data['y']
        return pd.DataFrame({'Vx': vx, 'Vy': vy, 'Rep': rep, 'Block': block,
                             'Time': time, 'Condition': condition, 'ID': id, 'pos x':pos_x, 'pos y':pos_y})

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
class Preprocessor(object):

    ###
    # This is a Preprocessing entity which can filter,
    # cut-via-threshold and plot DataFrames
    ###

    raw_paths = attr.ib(default='data')
    sample_rate = attr.ib(default=240)
    raw_headers = attr.ib(default=['SampleNum', 'x', 'y', 'z',
                                   'phi', 'theta', 'psi', 'Time', 'Event'])

    def load_df_from_directory_gen(self, dir_path):
        ###
        # This is a generator which takes csv
        # files from dir_path and yields Trials.
        ###

        assert Path(dir_path).is_dir(), \
            f'Destination directory does not exist ' \
            f'or is not a directory: {dir_path}'

        self.raw_paths = glob(str(Path(dir_path).joinpath('li_*_*_*_*.csv')))
        assert len(self.raw_paths) > 0, f'No source files found!'

        trial_out = Trial()
        for fn in self.raw_paths:
            try:
                df = pd.read_csv(fn, names=self.raw_headers)
                df = df.set_index('Time')
                trial_out.position_data = df[['x', 'y', 'z']]
                trial_out.events = df[['Event']]
                trial_out.velocity_data = df[['x', 'y', 'z']].diff()
                trial_data = os.path.split(fn)[1].split(sep='_')
                trial_out.stimulus = trial_data[1] + '_' + trial_data[2]
                trial_out.block = int(trial_data[3])
                trial_out.rep = int(os.path.splitext(trial_data[4])[0])
                trial_out.raw_file_path = fn
                trial_out.id = os.path.split(dir_path)[1]
                yield trial_out
            except ValueError:
                raise AssertionError(f'Could not load {fn}.')

    def load_single_file(self, file_path):
        ###
        # This method loads a single csv and yields a single Trial
        ###

        assert Path(file_path).is_file(), f'File does not exists: {file_path}'
        file_path = Path(file_path)

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
    def plot(data_in: pd.DataFrame):
        ###
        # This is a plotting method that will work on a given DataFrame
        # mode = 'pandas' will plot using Pandas and matplotlib.
        ###

        assert isinstance(data_in, pd.DataFrame)
        assert not data_in.empty, f'No data to plot!'

        plt.figure()
        data_in.plot()
        plt.show()

    @staticmethod
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def filter_raw_data(self, data_in: pd.DataFrame,
                        lpf_on=True):
        ###
        # This method applies the Savitsky-Golay filter
        # on a data frame.
        # For example, the velocity is extracted by
        # setting deriv=1 and data_in to position data.
        #
        # S-G paramters:
        # window_len - how many samples are used for
        # polynomial fitting of order polyorder.
        # deriv=n controls whether we smooth the data
        # or its n'th derivative.
        ###

        assert isinstance(data_in, pd.DataFrame)
        assert not data_in.empty, f'No data to process!'

        df = data_in.copy()

        # we start by filling NaNs in the data
        df = df.fillna(method='bfill')

        if lpf_on:
            # 5hz low pass (zero phase)
            cutoff = 5
            order = 2

            for col in df:
                df[col] = self.butter_lowpass_filter(df[col],
                                                     cutoff=cutoff,
                                                     fs=self.sample_rate,
                                                     order=order)
        return df

    @staticmethod
    def remove_baseline(data_in: pd.DataFrame, threshold=0.005):
        ###
        # This method takes a data frame of velocity data,
        # calculates the normalized magnitude of
        # tangential velocity and filters out any data
        # that's lower than a given threshold.
        ###

        assert isinstance(data_in, pd.DataFrame)
        assert not data_in.empty, f'No data to process!'
        assert threshold > 0, f'Threshold must be greater or equal than zero.'

        # calculate absolute velocity
        df = np.power(data_in, 2)
        df = df.sum(axis=1)

        # min-max normalization
        df = (df - df.min()) / (df.max() - df.min())

        # find data above threshold
        idx = df.loc[df >= threshold]

        # set data cutting limits
        low_cut_index = idx.index.min()-0.1 \
            if df.index.min() < idx.index.min()-0.1 \
            else df.index.min()
        high_cut_index = idx.index.max()+0.1 \
            if df.index.max() > idx.index.max()+0.1 \
            else df.index.max()

        return data_in.copy()[low_cut_index:high_cut_index]

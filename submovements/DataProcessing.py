import pandas as pd
import attr
from glob import glob
import os
import pandas as pd
import pandas.io.common
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool

@attr.s
class Preprocessor(object):
    raw_paths = attr.ib(default='data')
    database_file_path = attr.ib(default='data\\database.mat')
    sample_rate = attr.ib(default=240)

    def load_df_from_directory_gen(self, dir_path):
        headers = ['SampleNum', 'x_cm', 'y_cm', 'z_cm', 'phi', 'theta', 'psi', 'Time', 'Event']
        self.raw_paths = glob(os.path.join(dir_path, '*.csv'))
        for fn in self.raw_paths:
            try:
                raw_data = pd.read_csv(fn, names=headers)
                file_name = os.path.split(fn)[1]
                yield raw_data, file_name
            except IOError:
                print(f'{fn} was not loaded.')
                continue

    def plot(self, data_in: pd.DataFrame, mode='bokeh', sample_factor: int = 10):
        ###
        # mode = 'pandas' will plot using Pandas and matplotlib.
        # mode = 'bokeh' will plot using bokeh.
        # sample_factor dictates how many samples will be displayed (lower -> higher performance -> lower resolution).
        ###

        if mode=='pandas':
            data_in.plot()
        elif mode=='bokeh':
            f = figure()
            sample = data_in.sample(sample_factor)
            source = ColumnDataSource(sample)
            f.vline_stack(['y', 'x'], x='Time',
                     source=source)
            show(f)
        else:
            print(f'Mode {mode} is not supported.')
            return None


    def cut_away_baseline(self, data_in:pd.DataFrame):
        pass

    def filter_raw_data(self, data_in:pd.DataFrame, low_cutoff, high_cutoff):
        fs = self.sample_rate


        pass

    def save_as_mat(self, data_in:pd.DataFrame, destination):
        self.database_file_path = destination


        pass



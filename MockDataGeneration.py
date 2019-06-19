import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(np.random.rand(100, 5)).rename(columns={0: 'a', 1: 'b', 2: 'c'}).set_index(['a', 'b', 'c'])

df

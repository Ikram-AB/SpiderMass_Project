import os
import subprocess
import pandas as pd
import numpy as np
from pyopenms import *
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess_data(all_data):
    df1 = all_data.iloc[:, :4]
    df2 = all_data.T.iloc[4:]
    df2 = df2.fillna(df2.min(axis=0), axis=0)
    df2 = df2.T
    dfs = [df1['Class'], df1['File'], df1['RT'], df1['Sum'], df2]
    df_concatenated = pd.concat(dfs, axis=1)
    df_concatenated.iloc[:, 4:] = df_concatenated.iloc[:, 4:].div(df_concatenated['Sum'], axis=0)
    return df_concatenated
import pandas as pd
import numpy as np
from pyopenms import *
from scipy.signal import find_peaks

def load_data(file_path):
    all_data = pd.DataFrame()
    exp = MSExperiment()
    MzMLFile().load(file_path, exp)

    chromatogram = exp.getChromatograms()[0]
    times, intensities = chromatogram.get_peaks()

    summ = sum(intensities)
    h = 0.04 * summ
    peaks, _ = find_peaks(intensities, height=h)

    spectra = exp.getSpectra()
    pic_spectra = [spectra[peak_index] for peak_index in peaks]

    all_mz_values = set()
    for spectrum in pic_spectra:
        mz_values, _ = spectrum.get_peaks()
        all_mz_values.update(mz_values)

    all_mz_values = sorted(all_mz_values)
    rts = []
    sums = []
    data = []

    for spectrum in pic_spectra:
        mz_values, intensities = spectrum.get_peaks()
        rt_value = spectrum.getRT()
        rts.append(rt_value)
        intensity_dict = dict(zip(mz_values, intensities))
        row_data = {mz: intensity_dict.get(mz, np.nan) for mz in all_mz_values}
        row_data['File'] = os.path.basename(file_path)
        row_data['RT'] = rt_value
        row_data['Sum'] = sum(intensities)
        data.append(row_data)
    
    file_data = pd.DataFrame(data)
    all_data = pd.concat([all_data, file_data], ignore_index=True)

    desired_columns = ['File', 'RT', 'Sum'] + sorted(all_mz_values)
    all_data = all_data.reindex(columns=desired_columns)
    return all_data

def preprocess_data(all_data):
    df1 = all_data.iloc[:, :3]
    df2 = all_data.T.iloc[3:]
    df2 = df2.fillna(df2.min(axis=0), axis=0)
    df2 = df2.T
    dfs = [df1['File'], df1['RT'], df1['Sum'], df2]
    df_concatenated = pd.concat(dfs, axis=1)
    df_concatenated.iloc[:, 3:] = df_concatenated.iloc[:, 3:].div(df_concatenated['Sum'], axis=0)
    return df_concatenated

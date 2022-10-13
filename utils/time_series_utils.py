import numpy as np
import obspy.signal.filter as filter
#import dtw as dtw
from utils.mass.mass_naive import *


def z_norm(input):
    """input is ndarray"""

    if input.ndim == 1:
        return (input - input.mean()) / input.std()
    elif input.ndim == 2:
        results = np.empty(input.shape)
        for i in range(input.shape[0]):
            results[i, :] = z_norm(input[i, :])
        return results
    else:
        return None



#
# def ts_ss_basic_eu(x, y):
#     #do similarity search, search the shorter one from the longer time series.
#     #in here the order doesn't matter
#     #will return the min distance
#
#     # inputs are numpy ndarray
#     if x.ndim != 1 or y.ndim != 1:
#         print("x and y are not 1 dimensions")
#         return None
#
#     elif (x.shape[0] == 0 or y.shape[0] == 0):
#         print("x or y are empty")
#         return None
#
#     return mass_NN.mass_NN(x, y)
#
# def ts_ss_complete_eu(x, y, m):
#     """
#         x' of length m from x;
#         y' of length m from y;
#         compare all possible pairs of x' and y' and find the most similar one.
#
#     :param x: a 1d-ndarray
#     :param y: a 1d-ndarray
#     :param m: length of subsequence
#     :return: The smallest distance among  all possible pairs of (x', y'), the distance is after normalization.
#     """
#     if m <= 0:
#         print("m has to be greater than 0")
#         return None
#     if x.ndim != 1 or y.ndim != 1:
#         print("x or y are not 1 dimension")
#         return None
#     elif x.shape[0] < m or y.shape[0] < m:
#         print("x or y length < m")
#         return None
#     global_min = np.finfo(np.double).max
#
#     #total n - m + 1 subsequence
#     for i in range(y.shape[0] - m + 1):
#         ym = y[i : i + m]
#         d = ts_ss_basic_eu(x, ym)
#         if d < global_min:
#             global_min = d
#
#     return global_min




def truncate(data, fs, oldStartTime, oldEndTime, newStartTime, newEndTime):
    if newStartTime < oldStartTime or newEndTime > oldEndTime:
        return None

    newStartIndex = (newStartTime - oldStartTime) * fs
    if data.ndim == 1:
        data_len = data.shape[0]
        newEndIndex = data_len - 1 - (oldEndTime - newEndTime) * fs
        return data[newStartIndex : newEndIndex + 1]
    elif data.ndim == 2:
        data_len = data.shape[1]
        newEndIndex = data_len - 1 - (oldEndTime - newEndTime) * fs
        return data[:, np.int(newStartIndex) : np.int(newEndIndex) + 1]
    else:
        return None

def downSample(data, old_fs, new_fs):
    if new_fs > old_fs or old_fs % new_fs != 0:
        return None
    step = old_fs / new_fs
    if data.ndim == 1:
        return data[0 : data.shape[0] : np.int(step)]
    elif data.ndim == 2:
        return data[:, 0: data.shape[1]: np.int(step)]

def time_series_bandpass(input, low, high, sample, order, zero_phase=True):
    input_arr = np.array(input)
    if input_arr.ndim == 1:
        return filter.bandpass(input_arr, low, high, sample, order, zero_phase)
    elif input_arr.ndim == 2:
        results = np.empty(input_arr.shape)
        for i in range(input_arr.shape[0]):
            results[i] = filter.bandpass(input_arr[i], low, high, sample, order, zero_phase)
        return results
    else:
        return None




import numpy as np
from multiprocessing import Pool
from multiprocessing import shared_memory
from sklearn import metrics
from matplotlib import pyplot as plt

import utils.dtw.dtw_collections
from utils import time_series_utils as tsu
import os
from mpl_toolkits.basemap import Basemap
from matplotlib import collections  as mc


class DistUtils:
    @staticmethod
    def dist(template_shape, template_data_type, shm_name, query_index, dist_func, dist_func_args=None):
        '''
        Compute the distance array of template[query_index] with all other time-series in the template
        :param dist_func_args:
        :param template_shape:
        :param template_data_type: nxm matrix, n is the number of time-series, m is the length of each time-series
        :param shm_name:
        :param query_index:
        :param dist_func:
        :return:
        '''
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        template = np.ndarray(template_shape, dtype=template_data_type, buffer=existing_shm.buf)

        results = np.zeros(template_shape[0])
        results[:] = np.nan
        query = template[query_index, :]
        for i in range(template.shape[0]):
            d = dist_func(template[i, :], query, dist_func_args)
            results[i] = d
        existing_shm.close()
        return results

    @staticmethod
    def dist_complete(template_shape, template_data_type, shm_name, query_index, dist_func, dist_func_args=None):
        '''
        Compute the distance array of template[query_index] with all other time-series in the template
        :param dist_func_args:
        :param template_shape:
        :param template_data_type: nxm matrix, n is the number of time-series, m is the length of each time-series
        :param shm_name:
        :param query_index:
        :param dist_func:
        :return:
        '''
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        template = np.ndarray(template_shape, dtype=template_data_type, buffer=existing_shm.buf)

        results = np.zeros(template_shape[0])
        results[:] = np.nan
        complete_results = []
        query = template[query_index, :]
        for i in range(template.shape[0]):
            d, d_complete = dist_func(template[i, :], query, dist_func_args)
            results[i] = d
            complete_results.append(d_complete)

        existing_shm.close()
        return results, complete_results

    @staticmethod
    def dist2(template_shape, template_data_type, shm_name, query, dist_func, dist_func_args):
        '''
        Compute the distance array of query with all other time-series in the template
        query shold be the same length as each time-series in the templates
        :param dist_func_args:
        :param template_shape:
        :param template_data_type:
        :param shm_name:
        :param query:
        :param dist_func:
        :return:
        '''
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        template = np.ndarray(template_shape, dtype=template_data_type, buffer=existing_shm.buf)

        results = np.zeros(template_shape[0])
        results[:] = np.nan
        for i in range(template.shape[0]):
            d = dist_func(template[i, :], query, dist_func_args)
            results[i] = d
        existing_shm.close()
        return results

    @staticmethod
    def sym_dm(dist_matrix, corr_based):
        n = dist_matrix.shape[0]
        m = dist_matrix.shape[1]
        results = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if i == j:
                    if corr_based:
                        results[i, j] = 1
                    else:
                        results[i, j] = 0
                else:
                    if corr_based:
                        results[i, j] = np.max([dist_matrix[i, j], dist_matrix[j, i]])
                    else:
                        results[i, j] = np.min([dist_matrix[i, j], dist_matrix[j, i]])
        return results


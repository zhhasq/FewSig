from utils.dtw.dtw_collections import dtw_range_search, dtw_search_complete_simple
from utils.time_series_utils import z_norm
from utils.dist_helper import *
from utils.mass.mass_collections import *
from tqdm import tqdm, trange

class MetricInfo:
    def __init__(self, name, dist_func, dist_func_complete, corr_based, data_norm, dist_metric_args, num_process, batch):
        self.name = name
        self.dist_func = dist_func
        self.corr_based = corr_based
        self.data_norm = data_norm
        self.dist_metric_args = dist_metric_args
        self.dist_func_complete = dist_func_complete
        self.num_process =num_process
        self.batch_size = batch

    def __str__(self):
        return self.name

    def get_name(self):
        return self.name

    def execute(self, data):
        num_process = self.num_process
        if self.data_norm:
            data = z_norm(data)

        results = []
        num_ts = data.shape[0]
        batch_size = self.batch_size

        for j in range(0, num_ts, batch_size):
            up = j + batch_size
            if up > num_ts:
                up = num_ts
            dm = self.compute_dm_batch(num_process, data, j, up)
            results.append(dm)

        # make dm as distance based
        dm = np.vstack(results)
        if self.corr_based:
            dm[dm < 0] = 0
            dm = 1 - dm

        # make it symmetry
        results = np.zeros(dm.shape)
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                results[i, j] = np.min([dm[i, j], dm[j, i]])
        return results

    def compute_dm_batch(self, num_process, data, start, end):
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        b = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        b[:] = data[:]
        # execute in batches, memory overflow
        pool = Pool(processes=num_process)

        results = []
        result_async = []
        for i in range(start, end):
            result_async.append(pool.apply_async(DistUtils.dist, args=(
                b.shape, b.dtype, shm.name, i, self.dist_func, self.dist_metric_args)))

        for i in trange(len(result_async), desc=f'Computing {self.name}'):
            # if i % 20 == 0:
            #     print(start + i)
            results.append(result_async[i].get())

        shm.close()
        shm.unlink()
        # print("shared memory released")
        # print("Finish computing waiting close")
        pool.close()
        # print("waiting joined")
        pool.join()
        # print("joined")
        return np.array(results)

    def execute_complete(self, data):
        if self.data_norm:
            data = z_norm(data)

        results = []
        results_complete = []
        num_ts = data.shape[0]
        batch_size = self.batch_size

        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        b = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        b[:] = data[:]
        # execute in batches, memory overflow
        pool = Pool(processes=self.num_process)

        for j in range(0, num_ts, batch_size):
            up = j + batch_size
            if up > num_ts:
                up = num_ts
            dm, dm_c = self.compute_dm_batch_complete(pool, b, shm.name, data, j, up)
            # dm is 2d matrix, dm_c is a list of matrix
            results.append(dm)
            results_complete.extend(dm_c)

        shm.close()
        shm.unlink()
        print("shared memory released")
        print("Finish computing waiting close")
        pool.close()
        print("waiting joined")
        pool.join()
        print("joined")

        # make dm as distance based
        dm = np.vstack(results)
        results_complete = np.array(results_complete)
        if self.corr_based:
            dm[dm < 0] = 0
            dm = 1 - dm

        # make it symmetry
        r = np.zeros(dm.shape)
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                r[i, j] = np.min([dm[i, j], dm[j, i]])
        return r, results_complete


    def compute_dm_batch_complete(self, pool, b, shm_name, data, start, end):

        results = []
        results_complete = []
        result_async = []
        for i in range(start, end):
            result_async.append(pool.apply_async(DistUtils.dist_complete, args=(
            b.shape, b.dtype, shm_name, i, self.dist_func_complete, self.dist_metric_args)))

        for i in range(len(result_async)):
            if i % 20 == 0:
                print(start + i)
            r, r_complete = result_async[i].get()
            # r is 1d, r_complete is a list of ndarray
            r_complete_m = np.array(r_complete)
            if self.corr_based:
                r_complete_m[r_complete_m < 0] = 0
                r_complete_m = 1 - r_complete_m
            results.append(r)
            results_complete.append(r_complete_m)

        return np.array(results), results_complete


class DTWSearchInfo(MetricInfo):
    def __init__(self, warp_band, data_norm, sampling_rate, arrival_index, ts_len,
                 query_start, query_end, search_start, search_end, num_process, batch):
        self.sample_rate = sampling_rate
        self._warp_band = warp_band
        name = f"DTW_Search-(S_{search_start}s_{search_end}s)-" + \
               f"(Q_{query_start}s_{query_end}s)-(norm_{data_norm})-(warp_{self._warp_band})"
        corr_based = False
        dist_func = dtw_range_search
        dist_func_complete = dtw_search_complete_simple
        'covert input time to index'
        query_start = arrival_index - int(query_start * sampling_rate)
        if query_start < 0:
            query_start = 0
        query_end = arrival_index + int(query_end * sampling_rate)
        if query_end >= ts_len:
            query_end = ts_len - 1

        search_start = arrival_index - int(search_start * sampling_rate)
        if search_start < 0:
            search_start = 0
        search_end = arrival_index + int(search_end * sampling_rate)
        if search_end >= ts_len:
            search_end = ts_len - 1

        args = {'warp_window': self._warp_band,
                'q_start': query_start,
                'q_end': query_end,
                'search_start': search_start,
                'search_end': search_end}

        super(DTWSearchInfo, self).__init__(name, dist_func, dist_func_complete, corr_based, data_norm, args,
                                            num_process, batch)


class CorrSearchInfo(MetricInfo):
    def __init__(self, data_norm, sampling_rate, arrival_index, ts_len,
                 query_start, query_end, search_start, search_end, num_process, batch):
        name = f"Corr-search-(S_{search_start}s_{search_end}s)-" + \
               f"(Q_{query_start}s_{query_end}s)-(norm_{data_norm})"
        corr_based = True
        dist_func = mass_range
        dist_func_complete = mass_range_complete
        self.sample_rate = sampling_rate
        'covert input time to index'
        query_start = arrival_index - int(query_start * sampling_rate)
        if query_start < 0:
            query_start = 0
        query_end = arrival_index + int(query_end * sampling_rate)
        if query_end >= ts_len:
            query_end = ts_len - 1

        search_start = arrival_index - int(search_start * sampling_rate)
        if search_start < 0:
            search_start = 0
        search_end = arrival_index + int(search_end * sampling_rate)
        if search_end >= ts_len:
            search_end = ts_len - 1

        mass = Mass3(search_end - 1 - search_start + 1, query_end - 1 - query_start + 1)
        args = {
            'fft_length': mass.fft_batch_size,
            'mean_length': mass.mean_batch_size,
            'corr_length': mass.corr_batch_size,
            'q_start': query_start,
            'q_end': query_end,
            'search_start': search_start,
            'search_end': search_end}

        super(CorrSearchInfo, self).__init__(name, dist_func, dist_func_complete, corr_based, data_norm, args, num_process, batch)


class EuSearchInfo(MetricInfo):
    def __init__(self, data_norm, sampling_rate, arrival_index, ts_len,
                 query_start, query_end, search_start, search_end, num_process, batch):
        name = f"Eu-search-(S_{search_start}s_{search_end}s)-" + \
               f"(Q_{query_start}s_{query_end}s)-(norm_{data_norm})"
        corr_based = False
        self.sample_rate = sampling_rate
        dist_func = mass_range_eu
        dist_func_complete = mass_range_complete_eu
        'covert input time to index'
        query_start = arrival_index - int(query_start * sampling_rate)
        if query_start < 0:
            query_start = 0
        query_end = arrival_index + int(query_end * sampling_rate)
        if query_end >= ts_len:
            query_end = ts_len - 1

        search_start = arrival_index - int(search_start * sampling_rate)
        if search_start < 0:
            search_start = 0
        search_end = arrival_index + int(search_end * sampling_rate)
        if search_end >= ts_len:
            search_end = ts_len - 1

        mass = Mass3(search_end - 1 - search_start + 1, query_end - 1 - query_start + 1)
        args = {
            'fft_length': mass.fft_batch_size,
            'mean_length': mass.mean_batch_size,
            'corr_length': mass.corr_batch_size,
            'q_start': query_start,
            'q_end': query_end,
            'search_start': search_start,
            'search_end': search_end}

        super(EuSearchInfo, self).__init__(name, dist_func, dist_func_complete, corr_based, data_norm, args, num_process, batch)

class EuInfo(MetricInfo):
    def __init__(self, data_norm, ts_len, num_process, batch):
        name = f"Eu-(norm_{data_norm})"
        corr_based = False
        dist_func = mass_range_eu
        dist_func_complete = None
        'covert input time to index'
        mass = Mass3(ts_len, ts_len)
        args = {
            'fft_length': mass.fft_batch_size,
            'mean_length': mass.mean_batch_size,
            'corr_length': mass.corr_batch_size,
            'q_start': 0,
            'q_end': ts_len-1,
            'search_start': 0,
            'search_end': ts_len-1}

        super(EuInfo, self).__init__(name, dist_func, dist_func_complete, corr_based, data_norm, args, num_process, batch)


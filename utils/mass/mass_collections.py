from utils.mass.mass3_pyfftw import Mass3
import numpy as np



def mass_range_eu2(x, y, args):
    '''
        Using a sub-sequence in y to search in a subsequence x
        :param x: long time series to search
        :param y: contains query
        :param args:
        :return: return max correlation and also the index
        '''
    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    query_start = args.get("q_start")
    query_end = args.get("q_end")

    # define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    r = mass.execute_eu(x[search_start: search_end], y[query_start: query_end])
    bsf_i = np.argmin(r) + search_start
    return r.min(), bsf_i

def mass_range(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x: long time series to search
    :param y: contains query
    :param args:
    :return:
    '''
    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    r = mass.execute(x[search_start : search_end], y[query_start : query_end])
    return r.max()

def mass_range_eu(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x: long time series to search
    :param y: contains query
    :param args:
    :return:
    '''
    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    r = mass.execute_eu(x[search_start : search_end], y[query_start : query_end])
    return r.min()

def mass_range_complete(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x: long time series to search
    :param y: contains query
    :param args:
    :return:
    '''
    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    r = mass.execute(x[search_start : search_end], y[query_start : query_end])
    return r.max(), r

def mass_range_complete_eu(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x: long time series to search
    :param y: contains query
    :param args:
    :return:
    '''
    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    r = mass.execute_eu(x[search_start : search_end], y[query_start : query_end])
    return r.min(), r

def mass_range2(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x: long time series to search
    :param y: contains query
    :param args:
    :return: return max correlation and also the index
    '''
    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    r = mass.execute(x[search_start : search_end], y[query_start : query_end])
    bsf_i = np.argmax(r) + search_start
    return r.max(), bsf_i

def mass_var_window(x, y, args):
    #mass_list = args.get("mass_list")
    index_range = args.get("index_range")

    # used for creating fftw plan
    fft_length = args.get("fft_length")
    mean_length = args.get("mean_length")
    corr_length = args.get("corr_length")

    #
    left_bound = args.get("left_bound")
    right_bound = args.get("right_bound")

    mass = Mass3(len(x), len(x), fft_length, mean_length, corr_length)
    corrs = np.zeros(len(index_range))
    for i in range(len(index_range)):
        r = index_range[i]
        left = r[0]
        right = r[1]
        l = left - left_bound
        if l < 0:
            l = 0
        r = right + right_bound
        if r >= len(x):
            r = len(x) - 1
        tmp1=mass.execute(x[l : r], y[left : right]).max()
        tmp2=mass.execute(y[l : r], x[left : right]).max()
        if tmp1 > tmp2:
            corrs[i] = tmp1
        else:
            corrs[i] = tmp2
    return corrs.max()
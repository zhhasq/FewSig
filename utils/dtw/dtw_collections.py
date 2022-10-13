import numpy as np
import utils.dtw.dtw_ucr as dtw_ucr
import utils.dtw.dtw_basic_search2 as dtw_basic_search_simple



def dtw_range_search(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x:
    :param y:
    :param args:
    :return:
    '''
    warp_window = args.get("warp_window")
    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")
    bsf_index = np.array([-1])
    return dtw_ucr.dtw_ucr(x[search_start: search_end], y[query_start: query_end], warp_window, bsf_index)[0]

def dtw_search_complete_simple(x, y, args):
    '''
    Using a sub-sequence in y to search in a subsequence x
    :param x:
    :param y:
    :param args:
    :return:
    '''
    warp_window = args.get("warp_window")
    query_start = args.get("q_start")
    query_end = args.get("q_end")

    #define the search range
    search_start = args.get("search_start")
    search_end = args.get("search_end")
    complete_dist = dtw_basic_search_simple_py(x[search_start: search_end], y[query_start: query_end], warp_window)
    return complete_dist.min(), complete_dist

def dtw_basic_search_simple_py(x, y, w):
    # will not compute the path and normalized distance
    m = len(y)
    if w == -1:
        w = m
    r = dtw_basic_search_simple.dtw_basic_search2(x, y, w)
    # result[:] is dtw distance,
    return r


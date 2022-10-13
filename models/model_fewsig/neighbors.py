import numpy as np

class Neighbors:
    def __init__(self, cur_index, dist_array, id_list=None, is_include_self=True):
        self._cur_index = cur_index
        self._dist_array = dist_array
        self._sort_dist_list = self.sort()
        self._id_list = id_list
        #if the distance array include the dist with itself
        self._is_include_self = is_include_self
        self._NN_map = None

    def sort(self):
        cur_dist = self._dist_array
        tmp = [(k, cur_dist[k]) for k in range(len(cur_dist))]
        tmp_sort = sorted(tmp, key=lambda x: x[1])
        return tmp_sort

    # def get_NN(self, num, limit=None):
    #     # only get NN that has id < limit
    #     results = []
    #     i = 0
    #     while i < len(self._sort_dist_list) and len(results) < num:
    #         if limit is None or self._sort_dist_list[i][0] < limit:
    #             if self._sort_dist_list[i][0] != self._cur_index or not self._is_include_self:
    #                 results.append(NNPair(self._cur_index, self._sort_dist_list[i][0], self._sort_dist_list[i][1],
    #                                           self._id_list))
    #         i += 1
    #     return results

    def get_NN(self, num, set=None, exclude_index=-1):
        results = []
        i = 0
        while i < len(self._sort_dist_list) and len(results) < num:
            if (set is None or self._sort_dist_list[i][0] in set) and self._sort_dist_list[i][0] != exclude_index:
                if not self._is_include_self or self._sort_dist_list[i][0] != self._cur_index:
                    results.append(NNPair(self._cur_index, self._sort_dist_list[i][0], self._sort_dist_list[i][1],
                                      self._id_list))
                else:
                    pass
            i += 1
        return results

    def get_pre_NN(self, num, set=None):
        # index for NN is before self._cur_index
        results = []
        i = 0
        while i < len(self._sort_dist_list) and len(results) < num:
            if set is None or self._sort_dist_list[i][0] in set:
                if self._sort_dist_list[i][0] < self._cur_index:
                    results.append(NNPair(self._cur_index, self._sort_dist_list[i][0], self._sort_dist_list[i][1],
                                          self._id_list))
            i += 1
        return results

    def get_NN_ave(self, num, set=None, exclude_index=-1):
        #return self.get_NN_max(num, set)
        tmp_results = self.get_NN(num, set, exclude_index)
        # if len(tmp_results) < num:
        #     return None
        if len(tmp_results) == 0:
            return None
        return NNPairAve(tmp_results, self._id_list)

    def get_pre_NN_ave(self, num, set=None):
        tmp_results = self.get_pre_NN(num, set)
        if len(tmp_results) < num:
            return None
        else:
            return NNPairAve(tmp_results, self._id_list)

    def get_NN_max(self, num, set=None):
        tmp_results = self.get_NN(num, set)
        if len(tmp_results) < num:
            return None
        return tmp_results[-1]

    def get_dist(self, target_index):
        '''
        Get distance with a specified object

        :return:
        '''
        if self._NN_map is None:
            self._NN_map = self._build_NN_map()
        return self._NN_map.get(target_index)


    def _build_NN_map(self):
        result = dict()
        for t in self._sort_dist_list:
            result[t[0]] = t[1]
        return result

    def get_intra_NN_ave(self, neighbors_map, template_index_set, num_nn):
        NN_pair_list = self.get_NN(num_nn, template_index_set)
        if len(NN_pair_list) < num_nn:
            return None
        return IntraNNPairAve(NN_pair_list, neighbors_map)



    # def get_NN(self, num, limit1=None, limit2=None):
    #     # only get NN that has id within [limit1, limit2)
    #     results = []
    #     i = 0
    #     while i < len(self._sort_dist_list) and len(results) < num:
    #         if limit2 is None or self._sort_dist_list[i][0] < limit2:
    #             if limit1 is None or self._sort_dist_list[i][0] >= limit1:
    #                 if self._sort_dist_list[i][0] != self._cur_index or not self._is_include_self:
    #                     results.append(NNPair(self._cur_index, self._sort_dist_list[i][0], self._sort_dist_list[i][1],
    #                                               self._id_list))
    #         i += 1
    #     return results

class NNPair:
    def __init__(self, cur_index, nn_index, dist, id_list = None):
        self.dist = dist
        self.cur_index = cur_index
        self.nn_index = nn_index
        if id_list is not None:
            self.cur_id = id_list[cur_index]
            self.nn_id = id_list[nn_index]

class NNPairAve:
    def __init__(self, nn_pair_list, id_list = None):
        self.cur_index = nn_pair_list[0].cur_index
        self.nn_index_list = [x.nn_index for x in nn_pair_list]
        self.dist_list = []
        dist = 0
        for cur_pair in nn_pair_list:
            dist += cur_pair.dist
            self.dist_list.append(cur_pair.dist)
        self.dist = dist / len(nn_pair_list)
        if id_list is not None:
            self.cur_id = id_list[self.cur_index]
            self.nn_id_list = [id_list[x] for x in self.nn_index_list]

class IntraNNPairAve:
    def __init__(self, NN_pair_list, neighbors_map):
        self.cur_index = NN_pair_list[0].cur_index
        self.nn_index_list = [x.nn_index for x in NN_pair_list]
        self.nn_intra_dists = []

        for cur_NN_index in self.nn_index_list:
            cur_neighbor = neighbors_map.get(cur_NN_index)
            for intra_NN_index in self.nn_index_list:
                if intra_NN_index != cur_NN_index:
                    self.nn_intra_dists.append(cur_neighbor.get_dist(intra_NN_index))
        self.nn_intra_dists = np.array(self.nn_intra_dists)
        if len(self.nn_intra_dists) < 1:
            self.dist = 0
        else:
            self.dist = self.nn_intra_dists.mean()


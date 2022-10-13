from sklearn.neighbors import KNeighborsClassifier


class ExpFDistKNN:
    def __init__(self, k):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k, metric='precomputed')

    def fit(self, data, labels):
        self.train_data = data
        self.train_labels = labels
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)


    def predict_prob(self, test_data):
        # only return the column for label as 1
        prob = self.model.predict_proba(test_data)
        return prob[:, 1]

    def get_neighbors_labels(self, test_data):
        nn_dist, nn_index = self.model.kneighbors(test_data, self.k)
        results = []
        for cur_nn in nn_index:
            tmp = []
            for cur_index in cur_nn:
                tmp.append(self.train_labels[cur_index])
            results.append(tmp)
        return results
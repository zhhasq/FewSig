import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from models.model_fewsig.NCFAE.NCFA_torch import NCFA


class NCFAKNN(BaseEstimator):
    def __init__(self, nc, k, nca_init_args, random_state=42):
        self.nc = nc
        self.k = k
        device = nca_init_args.get("device")
        loss_func_name = nca_init_args.get("loss_func_name")
        loss_args = nca_init_args.get("loss_args")
        num_iter = nca_init_args.get("num_iter")

        self.optimizer_name = nca_init_args.get("optimizer_name")
        self.optimizer_args = nca_init_args.get("optimizer_args")
        verbose = nca_init_args.get("verbose")

        self.nca_torch = NCFA(device, loss_func_name, loss_args, self.optimizer_name, self.optimizer_args,
                              dim=self.nc, init="identity", max_iters=num_iter, verbose=verbose)
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.is_fit = False
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device


    def fit(self, data, labels, weight=None):
        # print(f"Train NCA with nc={self.nc}, kNN with k={self.k}")
        data_torch = torch.from_numpy(data).double().to(self.device)
        label_torch = torch.from_numpy(labels).long().to(self.device)
        if weight is not None:
            weight_torch = torch.from_numpy(weight).double().to(self.device)
        else:
            weight_torch = None
        self.nca_torch.train(data_torch, label_torch, weight_torch)
        X_nca = self.nca_torch(data_torch).detach().cpu().numpy()
        self.knn.fit(X_nca, labels)
        self.is_fit = True

    def predict(self, data):
        data_torch = torch.from_numpy(data).double().to(self.device)
        data_nca = self.nca_torch(data_torch).detach().cpu().numpy()
        y_hat = self.knn.predict(data_nca)
        return y_hat

    def predict_train(self, data):
        'prevent knn overfitting'
        data_torch = torch.from_numpy(data).double().to(self.device)
        data_nca = self.nca_torch(data_torch).detach().cpu().numpy()
        dist, n_id  = self.knn.kneighbors(data_nca, n_neighbors= 2, return_distance=True)
        y = self.knn._y
        y_hat = [y[i[1]] for i in n_id]
        return np.array(y_hat)
    # def predict2(self, data):
    #     'also get the nn-index'
    #     y_hat = self.model.predict(data)
    #     data = self.model[0].transform(data)
    #     nn_index = self.model[1].kneighbors(data, return_distance=False)[0]
    #     y_hat2 = [self.model[1]._y[x] for x in nn_index]
    #     if y_hat2[0] != y_hat[0]:
    #         raise RuntimeError
    #
    #     return y_hat, nn_index
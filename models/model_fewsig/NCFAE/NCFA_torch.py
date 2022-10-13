import os

import numpy as np
import torch


class NCFA:
    """Neighbourhood Components Analysis [1].

    References:
      [1]: https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    """
    def __init__(self, device, loss_func_name, loss_args, optimizer_name, optimizer_args, dim=None, init="random",
                 max_iters=500, tol=1e-5, verbose=True):
        """Constructor.

        Args:
          dim (int): The dimension of the the learned
            linear map A. If no dimension is provided,
            we assume a square matrix A of same dimension
            as the input data. For small values of `dim`
            (i.e. 2, 3), NCA will perform dimensionality
            reduction.
          init (str): The type of initialization to use for
            the matrix A.
              - `random`: A = N(0, I)
              - `identity`: A = I
          max_iters (int): The maximum number of iterations
            to run the optimization for.
          tol (float): The tolerance for convergence. If the
            difference between consecutive solutions is within
            this number, optimization is terminated.
        """
        self.dim = dim
        self.init = init
        self.max_iters = max_iters
        self.tol = tol
        self._mean = None
        self._stddev = None
        self._losses = None
        self.device = device
        self.loss_func_name = loss_func_name
        self.loss_args = loss_args
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.verbose = verbose
        self.bsf_loss = None
        self.bsf_A = None
        if self.verbose:
            print(f"Successfully create NCA pytorch{os.linesep}Device:{self.device}{os.linesep}dim:{self.dim}{os.linesep}max_iter:{self.max_iters}{os.linesep}"
                f"tol:{self.tol}{os.linesep}"
                f"loss:{self.loss_func_name}{os.linesep}\tloss_args:{self.loss_args} {os.linesep}"
                f"optimizer:{self.optimizer_name}{os.linesep}\t optimizer args:{self.optimizer_args}{os.linesep}")

    def __call__(self, X):
        """Apply the learned linear map to the input.
        """
        with torch.no_grad():
            if self._mean is not None and self._stddev is not None:
                X = (X - self._mean) / self._stddev
            # return torch.mm(X, torch.t(self.A))
            return torch.mm(X, torch.t(self.bsf_A))

    def predict_prob(self, X):
        """Apply the learned linear map to the input.
        """
        if self._mean is not None and self._stddev is not None:
            X = (X - self._mean) / self._stddev
        return torch.mm(X, torch.t(self.A))

    def _init_transformation(self):
        """Initialize the linear transformation A.
        """
        if self.dim is None:
            self.dim = self.num_dims
        if self.init == "random":
            print('using random init')
            a = torch.randn(self.dim, self.num_dims, device=self.device) * 0.01
            self.A = torch.nn.Parameter(a)
        elif self.init == "identity":
            a = torch.eye(self.dim, self.num_dims, dtype=torch.double, device=self.device)
            self.A = torch.nn.Parameter(a)
        else:
            raise ValueError("[!] {} initialization is not supported.".format(self.init))

    @staticmethod
    def _pairwise_l2_sq(x):
        """Compute pairwise squared Euclidean distances.
        """
        dot = torch.mm(x.double(), torch.t(x.double()))
        norm_sq = torch.diag(dot)
        dist = norm_sq[None, :] - 2*dot + norm_sq[:, None]
        # dist = torch.clamp(dist, min=0)  # replace negative values with 0
        # I = torch.eye(dot.shape[0], dtype=torch.double, device=f"cuda:{dot.get_device()}")
        # I = torch.eye(dot.shape[0], dtype=torch.double, device=f"cpu")
        # dist = torch.sqrt(dist+I)
        # dist = dist - I
        dist = torch.clamp(dist, min=0)
        return dist.double()

    @staticmethod
    def _p_corr(x):
        p_corr = torch.corrcoef(x)
        return p_corr.double()

    @staticmethod
    def _softmax(x, norm, deduction_max):
        """Compute row-wise softmax.

        Notes:
          Since the input to this softmax is the negative of the
          pairwise L2 distances, we don't need to do the classical
          numerical stability trick.
        """
        if norm:
            m = x.mean(dim=1).reshape([x.shape[0], -1])
            stddev = x.std(unbiased=False, dim=1).reshape([x.shape[0], -1])
            x = (x - m) / stddev
            # m = x.mean(dim=0).reshape([-1, x.shape[0]])
            # stddev = x.std(unbiased=False, dim=0).reshape([-1, x.shape[0]])
            # x = (x - m) / stddev

        exp = torch.exp(x)
        s = exp.sum(dim=1).reshape([exp.shape[0],-1])
        return exp / s

    @property
    def mean(self):
        if self._mean is None:
            raise ValueError('No mean was computed. Make sure normalize is set to True.')
        return self._mean

    @property
    def stddev(self):
        if self._stddev is None:
            raise ValueError('No stddev was computed. Make sure normalize is set to True.')
        return self._stddev

    @property
    def losses(self):
        if self._losses is None:
            raise ValueError('There are no losses to report. You must call train first.')
        return self._losses

    def F1_loss(self, X, y_mask, true_labels, softmax_norm, softmax_de_max, softmax_neg_dist, dist_func):
        embedding = torch.mm(X, torch.t(self.A))
        if dist_func == "L2":
            distances = self._pairwise_l2_sq(embedding)
        elif dist_func == "PCorr":
            distances = self._p_corr(embedding)

        if softmax_neg_dist:
            p_ij = self._softmax(-distances, softmax_norm, softmax_de_max)
        else:
            p_ij = self._softmax(distances, softmax_norm, softmax_de_max)
        p_ij_mask = p_ij * y_mask.double()
        p_ij_diag = torch.diag(p_ij_mask)
        p_i = p_ij_mask.sum(dim=1)
        p_i = p_i - p_ij_diag

        true_labels_flip = 1-true_labels
        p_i_flip = 1 - p_i
        tp = (p_i * true_labels).sum()
        tn = (p_i * true_labels_flip).sum()
        fn = (p_i_flip * true_labels).sum()
        fp = (p_i_flip * true_labels_flip).sum()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_loss = -(2 * precision * recall) / (precision + recall)
        return f1_loss

    def Focal_loss(self, X, y_mask, softmax_norm, softmax_de_max, alpha, gamma, softmax_neg_dist, dist_func):
        embedding = torch.mm(X, torch.t(self.A))
        if dist_func == "L2":
            distances = self._pairwise_l2_sq(embedding)
        elif dist_func == "PCorr":
            distances = self._p_corr(embedding)

        if softmax_neg_dist:
            p_ij = self._softmax(-distances, softmax_norm, softmax_de_max)
        else:
            p_ij = self._softmax(distances, softmax_norm, softmax_de_max)
        p_ij_mask = p_ij * y_mask.double()
        p_ij_diag = torch.diag(p_ij_mask)
        p_i = p_ij_mask.sum(dim=1)
        p_i = p_i - p_ij_diag
        # p_i_non_zero = torch.masked_select(p_i, p_i != 0)
        # tmp =  torch.pow((1-p_i_non_zero), gamma)
        # tmp2 = torch.log(p_i_non_zero)

        tmp =  torch.pow((1-p_i), gamma)
        tmp2 = torch.log(p_i)

        classification_loss = (-alpha*tmp*tmp2).sum()
        loss = classification_loss
        return loss


    def Focal_loss_k(self, k, X, y_mask, softmax_norm, softmax_de_max, alpha, gamma, softmax_neg_dist, dist_func):
        embedding = torch.mm(X, torch.t(self.A))
        if dist_func == "L2":
            distances = self._pairwise_l2_sq(embedding)
        elif dist_func == "PCorr":
            distances = self._p_corr(embedding)

        if softmax_neg_dist:
            p_ij = self._softmax(-distances, softmax_norm, softmax_de_max)
        else:
            p_ij = self._softmax(distances, softmax_norm, softmax_de_max)
        p_ij_mask = p_ij * y_mask.double()
        p_ij_diag = torch.diag(p_ij_mask)
        I = torch.eye(p_ij.shape[0], dtype=torch.double, device=self.device)
        p_ij_mask2 = p_ij_mask - I*p_ij_diag
        p_ij_sort, indices = torch.sort(p_ij_mask2, dim=1, descending=True)
        #
        p_i = p_ij_sort[:, :k].sum(dim=1)
        # p_i = p_i - p_ij_diag
        # p_i_non_zero = torch.masked_select(p_i, p_i != 0)
        tmp =  torch.pow((1-p_i), gamma)
        tmp2 = torch.log(p_i)

        classification_loss = (-alpha*tmp*tmp2).sum()
        loss = classification_loss
        return loss


    def loss(self, X, y_mask):
        # compute pairwise squared Euclidean distances
        # in transformed space
        embedding = torch.mm(X, torch.t(self.A))
        distances = self._pairwise_l2_sq(embedding)

        'Modified by Sheng'
        # m = distances.mean(dim=1).reshape([distances.shape[0], -1])
        # stddev = distances.std(unbiased=False, dim=1).reshape([distances.shape[0], -1])
        # distances = (distances - m) / stddev

        # fill diagonal values such that exponentiating them
        # makes them equal to 0
        # distances.diagonal().copy_(np.inf*torch.ones(len(distances)))

        # compute pairwise probability matrix p_ij
        # defined by a softmax over negative squared
        # distances in the transformed space.
        # since we are dealing with negative values
        # with the largest value being 0, we need
        # not worry about numerical instabilities
        # in the softmax function
        'Modified by Sheng'
        # p_ij = self._softmax(-distances)
        p_ij = self._softmax(distances)
        # p_ij.diagonal().copy_(torch.zeros(len(distances)))
        # for each p_i, zero out any p_ij that
        # is not of the same class label as i
        p_ij_mask = p_ij * y_mask.double()

        # sum over js to compute p_i
        p_i = p_ij_mask.sum(dim=1)

        # compute expected number of points
        # correctly classified by summing
        # over all p_i's.
        # to maximize the above expectation
        # we can negate it and feed it to
        # a minimizer
        # for numerical stability, we only
        # log_sum over non-zero values
        classification_loss = -torch.log(torch.masked_select(p_i, p_i != 0)).sum()

        # to prevent the embeddings of different
        # classes from collapsing to the same
        # point, we add a hinge loss penalty
        'Modified by Sheng'
        # distances.diagonal().copy_(torch.zeros(len(distances)))
        margin_diff = (1 - distances) * (~y_mask).double()
        hinge_loss = torch.clamp(margin_diff, min=0).pow(2).sum(1).mean()

        # sum both loss terms and return
        # loss = classification_loss + hinge_loss
        loss = classification_loss
        return loss


    def train(
            self,
            X,
            y,
            weight=None,
            batch_size=None,
    ):
        """Trains NCA until convergence.

        Specifically, we maximize the expected number of points
        correctly classified under a stochastic selection rule.
        This rule is defined using a softmax over Euclidean distances
        in the transformed space.

        Args:
          X (torch.FloatTensor): The dataset of shape (N, D) where
            `D` is the dimension of the feature space and `N`
            is the number of training examples.
          y (torch.LongTensor): The class labels of shape (N,).
          batch_size (int): How many data samples to use in an SGD
            update step.
          lr (float): The learning rate.
          weight_decay (float): The strength of the L2 regularization
            on the learned transformation A.
          normalize (bool): Whether to whiten the input, i.e. to
            subtract the feature-wise mean and divide by the
            feature-wise standard deviation.
        """
        self._losses = []
        self.num_train, self.num_dims = X.shape
        # self.device = torch.device("cuda" if X.is_cuda else "cpu")
        if batch_size is None:
            batch_size = self.num_train
        batch_size = min(batch_size, self.num_train)

        # initialize the linear transformation matrix A
        self._init_transformation()

        # # zero-mean the input data
        # if normalize:
        #     self._mean = X.mean(dim=1).reshape([X.shape[0], 1])
        #     self._stddev = X.std(unbiased=False, dim=1).reshape([X.shape[0], 1])
        #     X = (X - self._mean) / self._stddev

        if self.optimizer_name == "SGD":
            optim_args = {
                'lr': self.optimizer_args.get("lr"),
            }
            if self.optimizer_args.get("weight_decay") is not None:
                optim_args['weight_decay'] = self.optimizer_args.get("weight_decay")
            if self.optimizer_args.get("momentum") is not None:
                optim_args['momentum'] = self.optimizer_args.get("momentum")
            optimizer = torch.optim.SGD([self.A], **optim_args)

        elif self.optimizer_name == "ADAGRAD":
            optim_args = {
                'lr': self.optimizer_args.get("lr"),
            }
            optimizer = torch.optim.Adagrad([self.A], **optim_args)
        elif self.optimizer_name == "ADAM":
            optim_args = {
                'lr': self.optimizer_args.get("lr"),
            }
            if self.optimizer_args.get("weight_decay") is not None:
                optim_args['weight_decay'] = self.optimizer_args.get("weight_decay")
            optimizer = torch.optim.Adam([self.A], **optim_args)

        iters_per_epoch = int(np.ceil(self.num_train / batch_size))
        i_global = 0
        for epoch in range(self.max_iters):
            rand_idxs = torch.randperm(len(y))  # shuffle dataset
            X = X[rand_idxs]
            y = y[rand_idxs]
            if weight is not None:
                weight = weight[rand_idxs]
            A_prev = optimizer.param_groups[0]['params'][0].clone()
            for i in range(iters_per_epoch):
                # grab batch
                X_batch = X[i*batch_size:(i+1)*batch_size]
                y_batch = y[i*batch_size:(i+1)*batch_size]
                if weight is not None:
                    w_batch = weight[i*batch_size:(i+1)*batch_size]
                # compute pairwise boolean class matrix
                y_mask = y_batch[:, None] == y_batch[None, :]
                'modified by Sheng'
                # compute loss and take gradient step
                optimizer.zero_grad()
                if self.loss_func_name == 'CE':
                    loss = self.loss(X_batch, y_mask)
                elif self.loss_func_name == "FOCAL":
                    if weight is not None:
                        alpha = w_batch
                    elif self.loss_args.get("alpha") is None:
                        alpha = y_batch * 1  + (1-y_batch) * 1
                    elif self.loss_args.get("alpha") == "ratio":
                        alpha = self.compute_alpha(y_batch)
                    else:
                        'Arg is a number'
                        tmp = self.loss_args.get("alpha")
                        alpha = y_batch * tmp  + (1-y_batch) * (1-tmp)
                    gamma = self.loss_args.get("gamma")
                    softmax_norm = self.loss_args.get("softmax_norm")
                    softmax_de_max = self.loss_args.get("softmax_de_max")
                    dist_func = self.loss_args.get("dist_func")

                    loss = self.Focal_loss(X_batch, y_mask, softmax_norm, softmax_de_max,
                                           alpha, gamma, self.loss_args.get("softmax_neg_dist"), dist_func)
                elif self.loss_func_name == "FOCAL_K":
                    if self.loss_args.get("alpha") is None:
                        alpha = y_batch * 1  + (1-y_batch) * 1
                    elif self.loss_args.get("alpha") == "ratio":
                        alpha = self.compute_alpha(y_batch)
                    else:
                        'Arg is a number'
                        tmp = self.loss_args.get("alpha")
                        alpha = y_batch * tmp  + (1-y_batch) * (1-tmp)
                    gamma = self.loss_args.get("gamma")
                    softmax_norm = self.loss_args.get("softmax_norm")
                    softmax_de_max = self.loss_args.get("softmax_de_max")
                    dist_func = self.loss_args.get("dist_func")
                    k = self.loss_args.get("k")
                    loss = self.Focal_loss_k(k, X_batch, y_mask, softmax_norm, softmax_de_max,
                                           alpha, gamma, self.loss_args.get("softmax_neg_dist"), dist_func)
                elif self.loss_func_name == "F1":
                    softmax_norm = self.loss_args.get("softmax_norm")
                    softmax_de_max = self.loss_args.get("softmax_de_max")
                    dist_func = self.loss_args.get("dist_func")
                    loss = self.F1_loss(X_batch, y_mask, y.double(), softmax_norm, softmax_de_max,
                                        self.loss_args.get("softmax_neg_dist"), dist_func)

                self._losses.append(loss.item())
                if self.bsf_loss is None or loss.item() < self.bsf_loss:
                    self.bsf_loss = loss.item()
                    self.bsf_A = torch.clone(self.A.detach())
                loss.backward()
                optimizer.step()

                i_global += 1
                if self.verbose:
                    if not i_global % 10:
                        print("epoch: {} - loss: {:.5f}".format(epoch+1, loss.item()))

            # check if within convergence
            # A_curr = optimizer.param_groups[0]['params'][0]
            # if torch.all(torch.abs(A_prev - A_curr) <= self.tol):
            #     print("[*] Optimization has converged in {} mini batch iterations.".format(i_global))
            #     print("epoch: {} - loss: {:.5f}".format(epoch + 1, loss.item()))
            #     break
        # print()
    def compute_alpha(self, y):
        num_positive = y.sum()
        num_negative = len(y) - num_positive
        p0 = 1 / (1 + num_negative/num_positive)
        p1 = 1 - p0
        return y * p1 + (1-y) * p0















import numpy as np
from random import sample
from psutil import virtual_memory


class SphericalKMeans():
    def __init__(self, k, initialization='random_samples', eps=1e-4, max_iter=100, n_init=5, logger=None, verbose=False, report_interval=10):
        self.initialization = initialization
        self.k = k
        self.eps = eps
        self.max_iter = max_iter
        self.n_init = n_init
        if logger is not None:
            self.logger = logger
        self.verbose = verbose
        self.report_interval = report_interval
        
    def print(self, msg, level='info'):
        if self.logger:
            print_func = getattr(self.logger, level)
        elif self.verbose:
            print_func = print
        else:
            return
        print_func(msg)

    def initialize(self, data):
        if self.initialization == 'random_uniform':
            self.print("initializing with centroids sampled uniformly from the unit hypersphere")
            self.centroids = unit_normalize(np.random.randn(self.k, self.d), inplace=True)
        elif self.initialization == 'random_samples':
            self.print("initializing with centroids sampled uniformly from the data set")
            self.centroids = unit_normalize(data[sample(list(range(data.shape[0])), k=self.k),:], inplace=True)
        elif self.initialization == 'spkmeans++':
            self.print("initializing with centroids sampled by the kmeans++ procedure, adapted to spherical data")
            self.centroids = unit_normalize(spKMeansPlusPlus(data, self.k), inplace=True)
        else:
            raise ValueError("initialization must be in {'random_uniform', 'random_samples', 'spkmeans++'}")

    def assign(self, data, _assignments=None):
        """note: _assignments is mostly used privately during fitting to avoid gc for potentially
        large assignment arrays; new assignments are written into this rather than being reallocated"""
        n = data.shape[0]
        free_mem = virtual_memory().free
        estimated_mem = data.dtype.itemsize * n * self.centroids.shape[0]

        if _assignments is None:
            assignments = np.empty(n, dtype=np.uint16)
        else:
            assignments = _assignments

        # the number of instances we can handle simultaneously for computing a fast data-centroid similarity matrix
        # (which is then argmaxed to get hard cluster assignments)
        chunk_size = min(n, int(n*(free_mem * 0.9)/estimated_mem))

        if chunk_size < n:
            self.print(
                """"warning: memory is scarce; assignment function is iterating with chunk size %d
                on dataset of size %d""" % (chunk_size, n), level='info')

        for i in range(0, n, chunk_size):
            i_upper = min(i+chunk_size, n)
            assignments[i:i_upper] = np.argmax(np.dot(data[i:i_upper, :], self.centroids.T), axis=1)

        return assignments

    def infer_centroids(self, data, assignments, _centroids=None):
        d = data.shape[1]

        if _centroids is None:
            centroids = np.empty(shape=(self.k, d))
        else:
            centroids = _centroids

        for i in range(self.k):
            centroids[i, :] = data[np.where(assignments==i)].sum(axis=0)

        centroids = unit_normalize(centroids, inplace=True)
        return centroids

    def fit(self, data):
        data = unit_normalize(data, inplace=False)

        n = data.shape[0]
        assignments = np.empty(n, dtype=np.uint16)
        results=[]

        for i in range(1, self.n_init+1):
            self.print("\nentering initialization %d\n" % i, level='info')
            assignments[:] = -1*np.ones(n, dtype=np.uint16)

            self.initialize(data)

            for j in range(1, self.max_iter+1):
                log = (j % self.report_interval == 0)
                if log:
                    self.print("entering iteration %d" % j, level='info')

                new_assignments = self.assign(data)
                centroids = self.infer_centroids(data, new_assignments, _centroids=self.centroids)

                # should always be true:
                # print("centroids is self.centroids: {}".format(centroids is self.centroids))

                proportion_changed = (assignments != new_assignments).sum()/n
                assignments = new_assignments

                if log:
                    self.print("%f changed on iteration %d" % (proportion_changed, j), level='info')

                if proportion_changed < self.eps:
                    self.print("only %f changed; exiting" % proportion_changed, level='info')
                    break

            objective = self.objective(data)
            self.print("objective is %f per data point after %d iterations" % (objective/n, j), level='info')

            results.append((objective, self.centroids.copy(), j))

        best_objective, best_centroids, iters = max(results)
        self.print("\nbest result has objective %f per data point after %d iterations" % (best_objective/n, iters), level='info')

        self.centroids = best_centroids

    def objective(self, data):
        assignments = self.assign(data)
        n = data.shape[0]
        return np.array([np.dot(self.centroids[assignments[i], :], data[i, :]) for i in range(n)]).sum()


def norm(a):
    return np.sqrt(np.square(a).sum(axis=-1))

def unit_normalize(data, inplace=False):
    norms = norm(data)
    # don't copy the data if you don't have to!
    if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-10):
        if not inplace:
            data = ((data.T) / norms).T
        else:
            _data = data.T
            _data /= norms
    return data

def spKMeansPlusPlus(data, k):
    """
    Initialize centers with the kmeans++ algorithm,adapted to directional data.
    data is assumed unit-normed
    """
    # sample a data point uniformly at random to start
    centers = np.empty((k, data.shape[1]))
    centers[0, :] = data[np.random.randint(data.shape[0]), :]
    min_dists = np.repeat(np.infty, data.shape[0])

    for i in range(1, k):
        # sequence of min distances from points to centers
        # first get cos similarities
        dists = (data * centers[i - 1, :]).sum(axis=1)
        # find squared distances on a projected hyperplane
        dists = np.clip(dists, -0.99999999, 0.99999999)
        # 2sin/(1+cos)
        dists = 2.0 * np.sqrt(1.0 - np.square(dists)) / (1.0 + dists)
        min_dists = np.minimum(min_dists, dists)
        cum_sum = np.cumsum(min_dists)
        idx = np.searchsorted(cum_sum, np.random.rand() * cum_sum[-1])
        centers[i, :] = data[idx, :]
    return centers


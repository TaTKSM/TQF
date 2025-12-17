import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from numba import jit
from joblib import Parallel, delayed



def energy_score(y: np.ndarray, zs: np.ndarray) -> float:
    """
    Compute Energy Score for an observed value (y) and ensemble forecasts (zs).
    """
    assert y.ndim == 1
    assert zs.ndim == 2
    assert y.shape[0] == zs.shape[1]
    a1 = np.linalg.norm(y - zs, axis=1).mean()
    a2 = pdist(zs).mean() if len(zs) > 1 else 0.

    return a1 - 0.5 * a2



@jit(nopython=True)
def energy_distance_sq_base(X: np.ndarray, Y: np.ndarray, wx: np.ndarray, wy: np.ndarray, 
                            interdist_X_is_given: bool = False, interdist_Y_is_given: bool = False) -> float:
    """
    Compute squared energy distance with given sample weights. Compiled with numba.
    """
    assert X.ndim == Y.ndim == 2
    m = len(X)
    n = len(Y)
    wx = wx / np.sum(wx)
    wy = wy / np.sum(wy)

    def kern(x: np.ndarray, y: np.ndarray):
        assert x.ndim == y.ndim == 1
        return - np.linalg.norm(x - y)

    out = 0.

    if interdist_X_is_given == False:
        for i in range(m):
            for j in range(m):
                out += kern(X[i], X[j]) * wx[i] * wx[j]

    if interdist_Y_is_given == False:
        for i in range(n):
            for j in range(n):
                out += kern(Y[i], Y[j]) * wy[i] * wy[j]

    for i in range(m):
        for j in range(n):
            out -= 2 * kern(X[i], Y[j]) * wx[i] * wy[j]

    out = max(out, 0.)    # this is the SQUARED energy distance

    return out



def energy_distance(X: np.ndarray, Y: np.ndarray, 
                    weight_X: np.ndarray | None = None, weight_Y: np.ndarray | None = None,
                    interdist_X: float | None = None, interdist_Y: float | None = None) -> float:
    """
    Compute energy distance with given sample weights.
    """
    assert (interdist_X is None) or (interdist_X > 0. and weight_X is None)
    assert (interdist_Y is None) or (interdist_Y > 0. and weight_Y is None)

    wx = np.ones(len(X)) if weight_X is None else weight_X
    wy = np.ones(len(Y)) if weight_Y is None else weight_Y
    
    ed_squared = energy_distance_sq_base(X=X, Y=Y, wx=wx, wy=wy, 
                                         interdist_X_is_given=False if interdist_X is None else True, 
                                         interdist_Y_is_given=False if interdist_Y is None else True)

    if interdist_X is not None:
        ed_squared -= interdist_X
    if interdist_Y is not None:
        ed_squared -= interdist_Y

    ed = np.sqrt(max(ed_squared, 0))

    return ed



def quantile(data: np.ndarray, q: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """
    Compute empirical quantiles with sample weights.
    """
    w = np.ones(len(data)) if weight is None else weight

    return quantile_base(data=data, q=q, weight=w)



@jit(nopython=True)
def quantile_base(data: np.ndarray, q: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Compute empirical quantiles with sample weights. Compiled with numba.
    """
    assert data.ndim == q.ndim == weight.ndim == 1
    N = data.shape[0]
    M = q.shape[0]
    assert len(weight) == N, "Lengths of data and w must be the same."
    if not np.all(weight >= 0):
        print(np.sort(weight)[:3])
    assert np.all(weight >= 0), "w must be non-negative."
    w = weight / np.sum(weight)
    assert np.all(q >= 0) and np.all(q <= 1), "q must be between 0 and 1."

    # --- Sort data and weights together ---
    idx = np.argsort(data)
    data = data[idx]
    w = w[idx]

    # --- Cumulative weights (Hazen midpoint shift) ---
    cum_w = np.empty(N, dtype=np.float64)
    cum_w[0] = w[0] / 2.0
    for i in range(1, N):
        cum_w[i] = cum_w[i - 1] + (w[i - 1] / 2.0) + (w[i] / 2.0)

    qmin = cum_w[0]
    qmax = cum_w[-1]

    results = np.empty(M, dtype=np.float64)

    for k in range(M):
        q_val = q[k]
        if q_val <= qmin:
            # lower extrapolation
            slope = (data[1] - data[0]) / (cum_w[1] - cum_w[0])
            results[k] = data[0] - slope * (qmin - q_val)
        elif q_val >= qmax:
            # upper extrapolation
            slope = (data[N - 1] - data[N - 2]) / (cum_w[N - 1] - cum_w[N - 2])
            results[k] = data[N - 1] + slope * (q_val - qmax)
        else:
            # search for insertion index
            i = np.searchsorted(cum_w, q_val) - 1
            if i < 0:
                i = 0
            elif i > N - 2:
                i = N - 2
            # linear interpolation
            f = (q_val - cum_w[i]) / (cum_w[i + 1] - cum_w[i])
            results[k] = (1.0 - f) * data[i] + f * data[i + 1]

    return results



@jit(nopython=True)
def custom_loss(V: np.ndarray, qs: np.ndarray, directions: np.ndarray, 
                pj_data: np.ndarray, weight: np.ndarray) -> float:
    """
    Given a point cloud (V) with weights (weight), compute its empirical quantiles (trial_vals)
    and return the discrepancy between trial_vals and given target quantiles (pj_data).
    """
    assert pj_data.shape == (len(qs), len(directions))
    D = directions.shape[1]
    assert V.ndim == 2
    assert V.shape[1] == D
    assert len(weight) == len(V), f"len(weight) = {len(weight)} != len(V) = {len(V)}"
    trial_vals = np.zeros(pj_data.shape)
    
    for i, vec in enumerate(directions):
        vec = np.ascontiguousarray(vec)
        trial_vals[:, i] = quantile_base(data=V @ vec, q=qs, weight=weight)
    
    loss = np.abs(trial_vals - pj_data).mean()

    return loss



def resample_kde(data: np.ndarray, n: int, seed: int, bw_adjust: float = 1/3,
                 weight: np.ndarray | None = None) -> np.ndarray:
    """
    Fit KDE to weighted point cloud data and sample new points from it. 
    """
    if weight is not None:
        assert len(data) == len(weight), f"len(data) = {len(data)} != len(weights) = {len(weight)}."
        w = weight / np.sum(weight) 
    else:
        w = None
    kde = gaussian_kde(dataset=data.T, weights=w)

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(data)
    _distances, indices = nbrs.kneighbors(data)
    vs = data - data[indices[:, 2]]    # vectors from each sample to its second nearest neighbor
    factor_ = np.mean([(v @ np.linalg.inv(kde.covariance) @ v) ** 0.5 for v in vs]) * bw_adjust
    kde.set_bandwidth(kde.factor * factor_)

    if n < 2000:
        X = kde.resample(size=2000, seed=seed).T
        X = KMeans(n_clusters=n, random_state=seed).fit(X=X).cluster_centers_
    else:
        X = kde.resample(size=n, seed=seed).T

    return X



def assign_weight(X_old: np.ndarray, X_new: np.ndarray, w_old: np.ndarray) -> np.ndarray:
    """
    Solve a weight assignment problem from an old point cloud (X_old) to a new one (X_new).
    """
    assert X_old.shape == X_new.shape
    C = pairwise_distances(X_old, X_new)
    row_ind, col_ind = linear_sum_assignment(C)
    w_new = np.zeros_like(w_old)
    
    for old_idx, new_idx in zip(row_ind, col_ind):
        w_new[new_idx] = w_old[old_idx]

    return w_new



def QMEM(qs: np.ndarray, directions: np.ndarray, pj_data: np.ndarray, 
         seed: int, n_jobs: int, n_ensemble: int, n_base: int, verbose: bool = True,
         bw_adjust: float = 1/3, ftol: float = 1e-5, patience: int = 0) -> tuple:    
    """
    Quantile-Matching Empirical Measure (QMEM) algorithm.
    The goal is to construct a set of weighted points that are consistent with 
    the prespecified data of directional quantiles (pj_data).
    
    Parameters
    ----------
    qs: array (M,)
        Quantiles to be matched.
    directions: array (K, D)
        Collection of K unit vectors in R^D.
    pj_data: array (M, K)
        Target quantiles.
    n_base: int
        Number of support points within a single optimization run.
    n_ensemble: int
        Number of point clouds, each of size n_base, to be aggregated in the end.
    patience:
        Patience parameter used in the optimization of n_base support points.
        If loss does not decrease for more than patience steps, the optimization terminates.
        With larger patience, the process may take longer to reach convergence.
    bw_adjust: float
        Parameter of KDE.
    """
    assert pj_data.shape == (len(qs), len(directions))
    assert n_ensemble >= 1
    assert patience >= 0
    D = directions.shape[1]

    def printf(s: str, end: str = '\n'):
        if verbose:
            print(s, end=end)

    def culo(Ys: np.ndarray, weight: np.ndarray) -> float:

        return custom_loss(V=Ys, qs=qs, directions=directions, pj_data=pj_data, weight=weight)

    def get_w(Ys: np.ndarray, w0: np.ndarray | None = None) -> np.ndarray:
        
        w0 = np.ones(len(Ys)) / len(Ys) if w0 is None else w0
        assert len(Ys) == len(w0)

        sol = minimize(fun=lambda w: culo(Ys=Ys, weight=w), x0=w0,
                       method='SLSQP', bounds=[(0., None)] * len(Ys), options={'ftol': ftol},
                       constraints=[{'type': 'eq', 'fun': lambda w_: np.sum(w_) - 1.}])
        return sol

    N_0 = 3 ** D
    # Initial trial: nonconvex optimization
    sol = minimize(fun=lambda X: culo(Ys=X.reshape(-1, D), weight=np.ones(N_0)),
                   x0=np.random.default_rng(seed).normal(size=N_0 * D), method='SLSQP')
    printf("Initial search done.")
    y0 = sol.x.reshape(-1, D)
    history = [(np.ones(len(y0)) / len(y0), y0, sol.fun)]
    Y_support = resample_kde(data=y0, n=n_base, seed=seed, bw_adjust=bw_adjust)
    w_opt = None
    Y_support_opt = None
    counter = patience
    idx_iter = 0

    while True:

        if w_opt is None:
            sol = get_w(Ys=Y_support)
        else:
            sol = get_w(Ys=Y_support, w0=assign_weight(X_old=Y_support_opt, X_new=Y_support, w_old=w_opt))

        history.append((sol.x / np.sum(sol.x), Y_support, sol.fun))

        if sol.fun > np.min([h[2] for h in history]) and idx_iter > 0:
            counter -= 1
        else:
            printf("Optimization done.")
            counter = patience

        if counter < 0:
            break
        else:
            w_opt = sol.x
            Y_support_opt = np.copy(Y_support)
            # generate new support points
            Y_support = resample_kde(data=Y_support_opt, weight=w_opt, seed=seed, n=n_base, bw_adjust=bw_adjust)

        idx_iter += 1

    Y_pool = resample_kde(data=Y_support_opt, weight=w_opt, seed=seed, n=2000, bw_adjust=bw_adjust)
    
    def f(k: int):
        
        idx = np.random.default_rng(seed=seed + k).choice(a=range(len(Y_pool)), size=n_base)
        Y_support = Y_pool[idx]
        sol = get_w(Ys=Y_support, w0=assign_weight(X_old=Y_support_opt, X_new=Y_support, w_old=w_opt))
        
        return sol.x, Y_support

    printf("Parallel optimization in progress...", end=' ')
    res = Parallel(n_jobs=n_jobs)(delayed(f)(k) for k in range(n_ensemble))
    w_opt_all = np.concatenate([r_[0] for r_ in res]) / len(res)    # normalize the sum to 1
    Y_support_all = np.concatenate([r_[1] for r_ in res])
    printf("Done.")
    printf("Dropping unnecessary support points...", end=' ')
    idx = np.argsort(w_opt_all)[::-1]
    w_opt_all = w_opt_all[idx]
    Y_support_all = Y_support_all[idx]
    
    def drop_tail(w_: np.ndarray, i: int):

        assert i < len(w_)
        idx_ = np.arange(i + 1)
        ww_ = w_[idx_]
        ww_ /= np.sum(ww_)
        
        return ww_, idx_
    
    res = []
    search_set = list(range(n_base - 1, len(w_opt_all), 5))

    for i in search_set:

        ww_, idx_ = drop_tail(w_=w_opt_all, i=i)
        loss_ = culo(Ys=Y_support_all[idx_], weight=ww_)
        res.append(loss_)

    idxes = [j <= search_set[np.argmin(res)] for j in range(len(w_opt_all))]

    w_ = w_opt_all[idxes] / np.sum(w_opt_all[idxes])
    y_ = Y_support_all[idxes]
    
    printf("Done.")
    printf("Number of support points: {}, selection rate: {:.1f} %"\
           .format(len(w_), len(w_) / len(w_opt_all) * 100))

    history.append((w_, y_, float(np.min(res))))
    
    return history

    



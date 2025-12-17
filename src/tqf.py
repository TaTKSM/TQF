import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble._forest import ForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from . import qmem




def pair_dist(ys: np.array, seed: int) -> np.ndarray:
    """
    Compute pairwise distances of a point set (ys).
    """    
    N_max = 1000
    if len(ys) > N_max:
        ys = np.random.default_rng(seed).permutation(ys)[: N_max]

    from scipy.spatial.distance import pdist
    dists = pdist(ys[:, None]) if np.ndim(ys) == 1 else pdist(ys)

    return dists



def trans(ys: np.ndarray, T_rff: int | str | np.ndarray | list | None, seed: int) -> tuple:
    """
    Helper function for QRF++. Expand the dimension of a univariate target y to 2*T_rff + 1,
    where T_rff is the number of random Fourier features (RFF).
    """
    assert ys.ndim == 1

    if (T_rff == 0) or (T_rff is None):    # --> No RFF used
        
        return None, ys

    else:
        dists = pair_dist(ys, seed=seed)

        if T_rff == 'median':
            
            wm = np.median(dists)
            ws = [wm * 0.5, wm, wm * 2.0]
    
        elif isinstance(T_rff, int) and T_rff >= 1:

            ws = np.quantile(dists, q=(np.arange(T_rff) + 0.5) / T_rff)

        elif isinstance(T_rff, (list, np.ndarray)):
            
            ws = np.array(T_rff)

        else:
            raise ValueError
    
        ys_big = np.c_[ys, np.c_[[np.cos(ys / w) for w in ws]].T, 
                           np.c_[[np.sin(ys / w) for w in ws]].T]
        
        assert ys_big.shape == (len(ys), 2 * len(ws) + 1)
        ys_big = ys_big / np.std(ys_big, axis=0) * ys.std()

        return (np.quantile(dists, q=np.arange(0.1, 1, 0.1)), ws), ys_big



def train_regressor(X: np.ndarray, y: np.ndarray, seed: int, T_rff: int | str | np.ndarray | list | None,
                    n_jobs: int, qrf_params: dict | None = None, scaler_X = None) -> tuple:
    """
    Create and train a QRF++ (= enhanced quantile random forest) model.
    """
    assert X.ndim == 2 and y.ndim == 1, (X.ndim, y.ndim)
    scaler_X = FunctionTransformer() if scaler_X is None else clone(scaler_X)
    params_fixed = {
        'random_state': seed,
        'max_samples_leaf': None,
    }
    if qrf_params is not None:
        assert set(params_fixed.keys()).isdisjoint(qrf_params.keys()), \
            f"qrf_params contains keys that conflict with fixed params: {set(params_fixed.keys()) & set(qrf_params.keys())}"
        params_fixed.update(qrf_params)
        
    REGRESSOR = RandomForestQuantileRegressor
    model = make_pipeline(scaler_X, REGRESSOR(n_jobs=n_jobs, **params_fixed))
    ws, y_wide = trans(ys=y, seed=seed, T_rff=T_rff)
    model.fit(X=X, y=y_wide, randomforestquantileregressor__sparse_pickle=True)

    return ws, y_wide, model



class TQF:
    """
    Tomographic Quantile Forests.
    """
    def __init__(self, y_scaler_whole=StandardScaler(), y_scaler_1d=StandardScaler()):

        self.y_scaler_whole = FunctionTransformer() if y_scaler_whole is None else clone(y_scaler_whole)
        self.y_scaler_1d = FunctionTransformer() if y_scaler_1d is None else clone(y_scaler_1d)
        self.D = None
        self.q_predictor = None
        self.n_directions = None
        self.fitted = False
        self.X_long = None
        self.y_long = None
        self.y_long_transformed_wide = None
        self.y_ws = None
        self.n_features_X = None
        self.n_rot = None
        self.seed_for_fit = None
    
    @staticmethod
    def make_uniform_directions(dim: int, n: int, seed: int) -> np.ndarray:

        # generate unit vectors
        assert n < 2000
        a = np.random.default_rng(seed).normal(size=(2000, dim))
        a /= np.linalg.norm(a, axis=1).reshape(-1, 1)
        km = KMeans(n_clusters=n, random_state=seed).fit(a)
        v = km.cluster_centers_
        assert v.shape == (n, dim)
        v /= np.linalg.norm(v, axis=1).reshape(-1, 1)

        return v

    def expand_v(self, v: np.ndarray, n_rot: int) -> np.ndarray:

        assert n_rot >= 1
        v = np.asarray(v)
        
        if n_rot == 1:
            return v
        else:
            if v.ndim == 1:
                v = v.reshape(1, -1)
            D = v.shape[1]
            from scipy.stats import ortho_group
            assert self.seed_for_fit is not None
            Os = ortho_group.rvs(dim=D, random_state=self.seed_for_fit, size=n_rot - 1)\
                                 .reshape(n_rot - 1, D, D)    # fix seed
            v = np.concatenate([v @ O for O in np.concatenate([np.eye(D)[None, :], Os], axis=0)], axis=1).squeeze()

            return v

    def projective_augmentation(self, X_: np.ndarray, Y_: np.ndarray, seed: int, 
                                n_augment: int, n_rot: int) -> tuple:

        assert X_.ndim == Y_.ndim == 2 and len(X_) == len(Y_) and Y_.shape[1] > 1
        X_long = np.repeat(X_, n_augment, axis=0)
        Y_long = np.repeat(Y_, n_augment, axis=0)
        v = np.random.default_rng(seed).normal(size=Y_long.shape)
        v /= np.linalg.norm(v, axis=1).reshape(-1, 1)
        X_long = np.c_[X_long, self.expand_v(v, n_rot=n_rot)]
        y_long = np.sum(Y_long * v, axis=1)
        assert y_long.shape == (len(Y_) * n_augment,)

        return X_long, y_long

    def fit(self, X: np.ndarray, Y: np.ndarray, seed: int, n_augment: int, n_rot: int,
            n_jobs: int, qrf_params: dict, T_rff: int | str | np.ndarray | list | None):
        
        assert X.ndim == Y.ndim == 2 and Y.shape[1] > 1
        self.n_rot = n_rot
        self.D = Y.shape[1]
        self.n_features_X = X.shape[1]
        self.seed_for_fit = seed

        Y_scaled = self.y_scaler_whole.fit_transform(Y)
        self.X_long, self.y_long = self.projective_augmentation(X_=X, Y_=Y_scaled, seed=seed, 
                                                                n_augment=n_augment, n_rot=self.n_rot)
        y_long_transformed = self.y_scaler_1d.fit_transform(self.y_long.reshape(-1, 1)).ravel()
        self.y_ws, self.y_long_transformed_wide, self.q_predictor = \
                             train_regressor(X=self.X_long, y=y_long_transformed, seed=seed, n_jobs=n_jobs,
                                             qrf_params=qrf_params, T_rff=T_rff)
        self.fitted = True

        return self
    
    def predict_q(self, X_long: np.ndarray, qs: np.ndarray | list) -> np.ndarray:

        assert self.fitted, "This model is not fitted yet."
        qs = np.asarray(qs)
        assert X_long.ndim == 2 and qs.ndim == 1

        u = self.q_predictor.predict(X_long, quantiles=qs.tolist())
        W = np.copy(X_long)
        W[:, self.n_features_X :] = - W[:, self.n_features_X :]
        v = - self.q_predictor.predict(W, quantiles=(1. - qs).tolist())
        out = 0.5 * (u + v)

        if out.ndim == 1:
            return out
        elif out.ndim == 3:
            # out.shape = (n_samples, n_outputs, n_quantiles)
            return out[:, 0, :]    # discard random fourier features
        else:
            if len(qs) == 1:
                # out.shape = (n_samples, n_outputs)
                return out[:, 0]    # discard random fourier features
            else:
                # out.shape = (n_samples, n_quantiles)
                return out

    def predict(self, seed: int, x: np.ndarray, n_qs: int, n_jobs: int, patience: int,
                n_ensemble: int, n_directions: int, verbose: bool = True) -> tuple:

        assert self.fitted, "This model is not fitted yet."
        assert x.ndim == 1
        assert n_qs > 4 and n_directions > 4

        qs = np.linspace(0, 1, n_qs + 1)[:-1] + 0.5 / n_qs
        self.n_directions = n_directions
        v = self.make_uniform_directions(dim=self.D, n=self.n_directions, seed=seed)
        x_long = np.c_[  np.tile(x, (self.n_directions, 1)), self.expand_v(v, n_rot=self.n_rot)  ]
        pj_data = self.predict_q(X_long=x_long, qs=qs).T
        assert pj_data.shape == (n_qs, self.n_directions)
        pj_data = self.y_scaler_1d.inverse_transform(pj_data.reshape(-1, 1))\
                      .reshape(n_qs, self.n_directions)
        
        hist_ = qmem.QMEM(qs=qs, directions=v, pj_data=pj_data, seed=seed, verbose=verbose,
                          n_jobs=n_jobs, n_ensemble=n_ensemble, n_base=100, patience=patience)
        
        wopt = hist_[-1][0]
        points = hist_[-1][1]
        points = self.y_scaler_whole.inverse_transform(points)

        return wopt, points



def target_importance_rf(forest: ForestRegressor, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the total "split gain (reduction in SSE)" per output.
    Output is the relative importance values of size (Y.shape[1],). 
    The sum is normalized to 1.
    """
    assert Y.ndim == 2, "Y must be shape (n_samples, n_targets)"
    assert forest.n_outputs_ == Y.shape[1]
    raw_all = np.zeros((len(forest.estimators_), Y.shape[1]), dtype=float)

    for k, est in enumerate(forest.estimators_):

        raw = np.zeros(Y.shape[1], dtype=float)
        tree = est.tree_
        # binary data about whether each sample passed through each node or not
        ind = est.decision_path(X)  # shape (n_samples, n_nodes), sparse CSR

        feature = tree.feature
        threshold = tree.threshold

        for node in range(tree.node_count):

            if tree.children_left[node] == -1:  # leaf
                continue

            # samples that go through the parent node
            idx_parent = ind[:, node].toarray().ravel().astype(bool)

            # feature and threshold used at each node
            f = feature[node]
            thr = threshold[node]
            # split data
            idx_left = idx_parent & (X[:, f] <= thr)
            idx_right = idx_parent & (X[:, f] > thr)

            # compute SSE = sum (y - mean)^2 for each target
            Yp = Y[idx_parent]
            Yl = Y[idx_left]
            Yr = Y[idx_right]

            if len(Yp) == 0:
                continue
            else:
                mu_p = Yp.mean(axis=0)
                sse_p = ((Yp - mu_p)**2).sum(axis=0)

                if len(Yl) > 0:
                    mu_l = Yl.mean(axis=0)
                    sse_l = ((Yl - mu_l)**2).sum(axis=0)
                else:
                    sse_l = 0    
                
                if len(Yr) > 0:
                    mu_r = Yr.mean(axis=0)
                    sse_r = ((Yr - mu_r)**2).sum(axis=0)
                else:
                    sse_r = 0

                raw += (sse_p - (sse_l + sse_r))

        raw = np.maximum(raw, 0.0)
        relative_values = raw / raw.sum() if raw.sum() > 0 else raw
        raw_all[k] = relative_values
    
    return raw_all.mean(axis=0), raw_all.std(axis=0)



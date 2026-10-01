"""Game-block permutation e-process for the in-season alarm."""
import numpy as np
D = 8

def log_ll_Q(X, Q):               # log L_Q over consecutive pairs, per row; X (R,n)
    return np.log(Q[X[:, :-1], X[:, 1:]]).sum(1)

def game_eprocess(x, games, B, rng, lam=0.5):
    """Block permutation e-process, one block per outing. Q is fit on all earlier outings.
    Within an outing the pitches are permuted; the block e-value is (B+1)L_Q(X)/sum_b L_Q(X_b),
    which has conditional mean exactly 1 given the past whenever the outing is exchangeable
    given the past. The iid-marginal likelihood is permutation invariant and cancels."""
    logE, path = 0.0, []
    ug, start = np.unique(games, return_index=True)
    order = np.argsort(start); starts = start[order]
    bounds = list(starts) + [len(x)]
    N = np.zeros((D, D))
    for k in range(len(starts)):
        s, e = bounds[k], bounds[k + 1]
        blk = x[s:e]
        if k > 0 and len(blk) >= 3:
            Q = (N + lam) / (N.sum(1, keepdims=True) + lam * D)
            perm = np.argsort(rng.random((B, len(blk))), axis=1)
            Xb = np.vstack([blk[None, :], blk[perm]])
            ll = log_ll_Q(Xb, Q)
            m = ll.max()
            loge = np.log(B + 1) + ll[0] - (m + np.log(np.exp(ll - m).sum()))
            logE += loge
        path.append(logE)
        np.add.at(N, (blk[:-1], blk[1:]), 1)
    return np.array(path)

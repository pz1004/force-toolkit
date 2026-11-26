"""Statistical estimability is separate from snapshot availability."""
import numpy as np

PAIR_STATUS_LABELS = {
    0: 'estimable', 1: 'insufficient_observations',
    2: 'zero_variance', 3: 'diagonal_not_applicable',
}


def estimability(codes):
    if codes is None:
        return {'any_pair_estimable': False, 'all_pairs_estimable': False}
    off_diagonal = ~np.eye(len(codes), dtype=bool)
    valid = np.asarray(codes)[off_diagonal] == 0
    return {'any_pair_estimable': bool(np.any(valid)),
            'all_pairs_estimable': bool(valid.size and np.all(valid))}


def streaming_mad_codes(moments):
    p = moments.p
    codes = np.full((p, p), 1, dtype=np.int8)
    j, k = np.triu_indices(p, 1)
    pair = np.where(moments.count[j, k] < 2, 1,
                    np.where((moments.m2_j[j, k] > 0) & (moments.m2_k[j, k] > 0), 0, 2))
    codes[j, k] = codes[k, j] = pair
    np.fill_diagonal(codes, 3)
    return codes

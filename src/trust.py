import numpy as np

def aggregate(updates, counts, trust=None, max_samples=100000, floor=0.1):
    counts = np.clip(np.asarray(counts, dtype=float), 0, max_samples)
    if trust is None: weights = counts
    else:
        # Invalid trust is not evidence of reliability; apply the minimum trust.
        safe_trust = np.nan_to_num(np.asarray(trust, dtype=float), nan=floor, posinf=floor, neginf=floor)
        weights = counts * np.clip(safe_trust, floor, 1.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0: weights = np.ones(len(updates))
    weights = weights / weights.sum()
    return np.average(np.asarray(updates), axis=0, weights=weights), weights

def indicators(updates):
    updates = np.asarray(updates); norms = np.linalg.norm(updates, axis=1)
    reference = np.median(updates, axis=0); refnorm = np.linalg.norm(reference)
    cos = np.zeros(len(updates))
    for i, u in enumerate(updates):
        if norms[i] > 1e-12 and refnorm > 1e-12: cos[i] = np.clip(np.dot(u, reference)/(norms[i]*refnorm), -1, 1)
        elif norms[i] <= 1e-12 and refnorm <= 1e-12: cos[i] = 1.0
    median = float(np.median(norms)); deviation = np.abs(np.log((norms+1e-12)/(median+1e-12)))
    return norms, cos, deviation

def score(previous, norm_deviation, cosine, loss_delta, cfg):
    norm_quality = np.exp(-norm_deviation / max(cfg.norm_threshold, 1e-9))
    direction_quality = np.clip((cosine-cfg.cosine_threshold)/(1-cfg.cosine_threshold), 0, 1)
    validation_quality = np.exp(-max(0, loss_delta)/max(cfg.validation_threshold, 1e-9))
    evidence = 0.25*norm_quality + 0.25*direction_quality + 0.50*validation_quality
    return float(np.clip(cfg.beta*previous+(1-cfg.beta)*evidence, cfg.trust_floor, 1)), float(evidence)

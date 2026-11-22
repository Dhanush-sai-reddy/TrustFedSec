import numpy as np

def flip_labels(y, fraction, rng):
    result = y.copy(); n = int(round(len(y) * np.clip(fraction, 0, 1)))
    if n: idx = rng.choice(len(y), n, replace=False); result[idx] = 1-result[idx]
    return result

def manipulate(update, strength): return -float(strength) * update

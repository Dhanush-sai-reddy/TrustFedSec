import numpy as np
import pytest
from config import Config
from src.dataset import prepare
from src.model import train
from src.coordinator import run
from src.trust import aggregate


def test_invalid_trust_and_zero_weights():
    updates = np.array([[1., 2.], [3., 4.], [5., 6.]])
    result, weights = aggregate(updates, [2, 2, 2], [np.nan, np.inf, -np.inf])
    assert np.isfinite(result).all()
    assert np.allclose(weights, [1/3]*3)
    _, weights = aggregate(updates, [0, 0, 0], [0, 0, 0])
    assert np.allclose(weights, [1/3]*3)
    assert weights.sum() == pytest.approx(1)


def test_sign_flip_weights_and_actual_model_match():
    cfg = Config(rounds=1)
    data = prepare(cfg)
    clean = run(data, cfg, 'Normal FedAvg')
    poisoned = run(data, cfg, 'Poisoned FedAvg')
    defended = run(data, cfg, 'TrustFed-Sec')
    rows = defended['clients']
    assert max(r['trust'] for r in rows[:2]) < min(r['trust'] for r in rows[2:])
    assert max(r['aggregation_weight'] for r in rows[:2]) < min(r['aggregation_weight'] for r in rows[2:])
    for result in (clean, poisoned, defended):
        assert sum(r['aggregation_weight'] for r in result['clients']) == pytest.approx(1)
    assert all(r['aggregation_weight'] == pytest.approx(.2) for r in poisoned['clients'])
    assert not np.allclose(clean['parameters'], poisoned['parameters'])
    initial = np.zeros(data['features'] + 1)
    updates = []
    for i, ids in enumerate(data['parts']):
        update = train(data['X_train'][ids], data['y_train'][ids], initial, cfg.epochs, cfg.learning_rate)
        updates.append(-cfg.strength * update if i < cfg.compromised else update)
    expected = np.sum(np.array(updates) * np.array([r['aggregation_weight'] for r in rows])[:, None], axis=0)
    assert np.allclose(expected, defended['parameters'])
    assert np.allclose(np.mean(updates, axis=0), poisoned['parameters'])


def test_unequal_sample_fedavg_and_trust_effect():
    updates = [[1., 0.], [0., 1.]]
    avg, weights = aggregate(updates, [1, 3])
    assert np.allclose(weights, [.25, .75])
    assert np.allclose(avg, [.25, .75])
    defended, weights = aggregate(updates, [1, 3], [1., .1])
    assert weights[1] < .75 and np.allclose(defended, weights)

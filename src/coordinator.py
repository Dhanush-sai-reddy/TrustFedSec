import time
import numpy as np
from dataclasses import asdict
from src.model import train
from src.attacks import flip_labels, manipulate
from src.trust import aggregate, indicators, score
from src.evaluation import evaluate

def run(data, cfg, scenario, callback=None):
    if scenario not in ('Normal FedAvg', 'Poisoned FedAvg', 'TrustFed-Sec'): raise ValueError('Unknown scenario')
    if not 0 <= cfg.compromised <= cfg.clients: raise ValueError('Invalid number of compromised clients')
    theta = np.zeros(data['features']+1); trusts = np.ones(cfg.clients); history=[]; clients=[]
    malicious = set(range(cfg.compromised)) if scenario != 'Normal FedAvg' else set()
    start = time.perf_counter()
    for r in range(cfg.rounds):
        updates=[]; counts=[]
        for i, ids in enumerate(data['parts']):
            y = data['y_train'][ids]
            if i in malicious and cfg.attack == 'label flip': y = flip_labels(y, cfg.poison_fraction, np.random.default_rng(np.random.SeedSequence([cfg.seed, r, i, 731])))
            update = train(data['X_train'][ids], y, theta, cfg.epochs, cfg.learning_rate)-theta
            if i in malicious and cfg.attack == 'sign flip': update = manipulate(update, cfg.strength)
            updates.append(update); counts.append(len(ids))
        norms, cos, dev = indicators(updates); baseline = evaluate(data['X_val'], data['y_val'], theta)['loss']; rows=[]
        for i, update in enumerate(updates):
            candidate = evaluate(data['X_val'], data['y_val'], theta+update); delta = candidate['loss']-baseline
            trusts[i], evidence = score(trusts[i], dev[i], cos[i], delta, cfg)
            rows.append({'round':r+1, 'client':i+1, 'samples':counts[i], 'update_norm':float(norms[i]), 'cosine':float(cos[i]), 'norm_deviation':float(dev[i]), 'validation_loss_delta':float(delta), 'candidate_attack_recall':candidate['attack_recall'], 'trust':float(trusts[i]), 'evidence':evidence, 'simulated_compromised':i in malicious, 'status':'Review' if dev[i]>cfg.norm_threshold or cos[i]<cfg.cosine_threshold or delta>cfg.validation_threshold else 'Normal'})
        delta, weights = aggregate(updates, counts, trusts if scenario == 'TrustFed-Sec' else None, cfg.max_samples, cfg.trust_floor)
        theta += delta
        for i, row in enumerate(rows): row['aggregation_weight'] = float(weights[i])
        clients.extend(rows)
        metrics=evaluate(data['X_val'], data['y_val'], theta); metrics['round']=r+1; history.append(metrics)
        if callback: callback(r+1, rows, metrics)
    elapsed=time.perf_counter()-start
    return {'scenario':scenario, 'config':asdict(cfg), 'source':data['source'], 'samples':data['samples'], 'features':data['features'], 'history':history, 'clients':clients, 'test':evaluate(data['X_test'],data['y_test'],theta), 'seconds':elapsed, 'communication_bytes':int(cfg.rounds*cfg.clients*2*theta.size*8), 'parameters':theta.tolist()}

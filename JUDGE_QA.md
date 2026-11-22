# Ten judge questions

1. **Why logistic regression?** Shared, interpretable parameter vectors make exact aggregation and CPU demonstrations straightforward; capacity is limited.
2. **What makes comparison fair?** Matching seed, split, client partitions, initial model, rounds and attack selection; the held-out test is shared. Later local updates differ as global models diverge.
3. **Can validation leak into training?** The preprocessing fit and client training use only training records; validation guides trust and is reused, so it is not an unbiased final evaluation. Test is separate.
4. **What if a benign client is non-IID?** Its update may appear anomalous and be down-weighted; smoothing and a trust floor limit but do not remove this risk.
5. **Why use the median direction?** It provides a simple robust reference, but a coordinated majority can move it and heterogeneous populations can make it misleading.
6. **How are attacks simulated?** Sign flipping scales a negative local update; label flipping changes a random fraction of compromised clients’ training labels. Neither is real malware.
7. **What does trust mean?** A bounded heuristic combining norm, direction, validation loss and history; it is not a calibrated probability of honesty.
8. **Can attackers evade it?** Yes. Adaptive updates can mimic benign statistics or exploit validation bias; no guarantee is claimed.
9. **What is communication overhead?** An estimate of full float64 parameter vectors sent to and returned from each client per round, excluding protocol costs.
10. **Is this privacy-preserving?** No. The central simulation sees updates and holds trusted validation data; there is no secure aggregation or differential privacy.

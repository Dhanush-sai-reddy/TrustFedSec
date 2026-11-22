# Five-minute live demonstration

1. **0:00–0:45 — Frame the problem.** Show that clients collaboratively train an intrusion detector and that a compromised update can distort the shared model. Explain that this is a local simulation.
2. **0:45–1:15 — Establish the data.** Point to the synthetic label, sample count, five clients, seed, IID split and held-out test. Do not call this UNSW-NB15.
3. **1:15–2:15 — Establish baseline and attack.** Run Normal FedAvg and Poisoned FedAvg with default sign flipping. Compare actual test metrics and validation curves. Explain that the outcome depends on the seed and configuration.
4. **2:15–3:30 — Run defense.** Run TrustFed-Sec with unchanged controls. Inspect a flagged client, its direction, validation loss impact, trust and weight. Explain that a flag is not proof of malice and the defense softly down-weights rather than immediately excludes.
5. **3:30–4:30 — Compare honestly.** Read accuracy, macro F1, attack recall and false positive rate. Say explicitly whether this run improved or degraded each result. Show communication estimate and runtime.
6. **4:30–5:00 — Limitations.** Discuss trusted validation, non-IID clients, collusion, adaptive attacks and this centralized simulation. Optionally change to label flipping and rerun all three; settings changes hide incompatible comparisons.
